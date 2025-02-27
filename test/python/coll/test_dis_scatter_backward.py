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
from ast import arg
import datetime
import os
import random
import time
from flux.testing.moe_utils import calc_gather_index
import flux.testing
import torch
import torch.distributed as dist
import flux
from flux.testing import (
    DTYPE_MAP,
    initialize_distributed,
    gate_func,
    moe_gather_rs_forward_torch,
    generate_data,
)
import numpy as np
from flux.testing.perf_db_helper import log_perf, set_global_args, should_log_to_rds
from flux.util import get_arch
from flux.testing.moe_utils import calc_scatter_index_stable


EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
np.random.seed(3)  # need the same seed for all ranks to generate the same token distribution
random.seed(3)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)


def calc_gather_index_stable(choosed_experts: torch.Tensor, topk, ntokens):
    _, index_choosed_experts = choosed_experts.flatten().sort(stable=True)
    gather_index = index_choosed_experts.to(torch.int32) // topk
    topk_index = torch.arange(0, topk, dtype=torch.int32, device="cuda").repeat(ntokens)[
        index_choosed_experts
    ]
    return gather_index, topk_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=16384)
    parser.add_argument("-N", type=int, default=6144)
    parser.add_argument("-G", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--iters", default=1, type=int, help="perf iterations")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--sm_margin", default=16, type=int, help="sm margin")
    parser.add_argument(
        "--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys())
    )
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument(
        "--local_world_size", type=int, default=None, help="params use to simulate multiple nodes"
    )
    return parser.parse_args()


def generate_random_exp_indices(token_num, total_num_experts, topk):
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for tid in range(token_num):
        top_selected = random.sample(exp_list, topk)
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


def perf_torch(input, exp_indices):
    assert exp_indices.size(1) == args.topk
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    # calculate the scatter idx

    gather_index, topk_index = calc_gather_index_stable(exp_indices, args.topk, exp_indices.size(0))
    new_index = args.topk * gather_index + topk_index
    # calculate the gather idx accordingly
    # following are the all2all
    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    ep_size = EP_GROUP.size()
    num_experts_per_rank = args.G // ep_size
    a2a_splits_cpu = a2a_splits.cpu()
    permute_a2a_splits_cpu = (
        a2a_splits_cpu.reshape(-1, num_experts_per_rank).permute(-1, -2).flatten()
    )

    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(args.iters):
        permute_a2a_expert_output_list = torch.split(input, permute_a2a_splits_cpu.tolist())
        # print(f"Len: {len(permute_a2a_expert_output_list)}")
        a2a_expert_output_list = list()
        for idy in range(ep_size):
            for idx in range(num_experts_per_rank):
                a2a_expert_output_list.append(permute_a2a_expert_output_list[idx * ep_size + idy])
        a2a_expert_output = torch.cat(a2a_expert_output_list, dim=0)
        all2all_out = torch.empty(
            [splits_cpu_cur_rank.sum(), input.shape[-1]], device=input.device, dtype=input.dtype
        )
        torch.distributed.all_to_all_single(
            output=all2all_out,
            input=a2a_expert_output,
            output_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(dim=-1).tolist(),
            input_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
            group=EP_GROUP,
        )
        gather_output = torch.empty_like(all2all_out)
        gather_output[new_index] = all2all_out
        topk_reduce = gather_output.view(
            (gather_output.size(0) // args.topk, args.topk, gather_output.size(-1))
        ).sum(1)
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return topk_reduce, avg_time


def perf_flux(input, exp_indices):
    # print(f"exp_indices: {exp_indices.size()}")
    n_token_cur_ep_rank = exp_indices.size(0)
    input_len = torch.tensor([n_token_cur_ep_rank], dtype=torch.int32, device=input.device)
    ag_input_len = torch.zeros(EP_GROUP.size(), dtype=torch.int32, device=input.device)
    torch.distributed.all_gather_into_tensor(ag_input_len, input_len, group=EP_GROUP)
    ag_input_len_cpu = ag_input_len.cpu()
    print(f"flux ag_input_len_cpu: {ag_input_len_cpu}")
    ag_input_len_list = ag_input_len_cpu.tolist()
    ag_indices_len = ag_input_len * args.topk
    padded_indices = torch.empty([args.M, args.topk], dtype=torch.int32, device=input.device)
    padded_indices[: exp_indices.size(0),] = exp_indices
    ag_padded_indices = [torch.empty_like(padded_indices) for _ in range(EP_GROUP.size())]
    # concat the exp_indices from all the rank
    torch.distributed.all_gather(ag_padded_indices, padded_indices, group=EP_GROUP)
    ag_indices = torch.concat(
        [t[: ag_input_len_list[i], :] for i, t in enumerate(ag_padded_indices)]
    )
    print(f"ag_idices size: {ag_indices.size()}")
    splits_gpu = torch.bincount(ag_indices.view(-1), minlength=args.G).to(torch.int32)
    splits_cpu = splits_gpu.cpu()
    print("splits_cpu: ", splits_cpu, splits_cpu.size())
    # ag_scatter_idx = calc_scatter_index(ag_indices, splits_gpu)
    # scatter_idx_cur_rank = calc_scatter_index(exp_indices, splits_gpu_cur_rank)
    # the scatter_idx of the ranks within the same node should be the same,
    # either use the stable version or use the non-stable version + a broadcast within the node
    ag_scatter_idx = calc_scatter_index_stable(ag_indices)
    global op
    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(args.iters):
        output = op.forward(
            input,
            ag_indices.view(-1),
            ag_scatter_idx.view(-1),
            splits_cpu,
            ag_input_len_cpu,
            args.sm_margin,
        )
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return output, avg_time


if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if args.local_world_size is not None:
        LOCAL_WORLD_SIZE = args.local_world_size
    RANK, WORLD_SIZE, NNODES = EP_GROUP.rank(), EP_GROUP.size(), flux.testing.NNODES()
    op = flux.DisScatterBackward(
        args.G, args.M, args.N, args.topk, RANK, 1, WORLD_SIZE, LOCAL_WORLD_SIZE
    )
    for rid in range(args.rounds):
        # random simulate token received from dataloader
        token_num = random.randint(args.M - 64, args.M)
        print(f"Received {token_num} tokens from dataloader @ rank:{RANK}")
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
        splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
        a2a_splits = torch.empty_like(splits_gpu_cur_rank)
        torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
        scatterd_size = a2a_splits.sum()
        input = torch.rand(scatterd_size, args.N, dtype=torch.bfloat16).to("cuda")
        print("input size:", input.size())
        with flux.group_profile(name="all2all_combine", group=EP_GROUP, do_prof=args.profile):
            ref_out, ref_time = perf_torch(input, exp_indices)
            print(ref_out)
            torch.cuda.synchronize()
            flux_out, flux_time = perf_flux(input, exp_indices)
        print(ref_out.size())
        print(flux_out.size())
        torch.cuda.synchronize()
        print(flux_out)
        print(ref_out)
        print(f"sum: torch:{torch.sum(ref_out)} flux:{torch.sum(flux_out)}")
        print(f"Flux: Time {flux_time} ms")
        print(f"Torch: Time {ref_time} ms")

        torch.distributed.barrier()
        flux.torch_allclose(flux_out, ref_out, 1e-2, 1e-2)
