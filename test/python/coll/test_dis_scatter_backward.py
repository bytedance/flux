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
import flux.testing
import torch
import torch.distributed as dist
import flux
from flux.testing import (
    DTYPE_MAP,
    initialize_distributed,
    split_torch_process_group,
    nvshmem_split_team_2d,
)
import numpy as np
from flux.testing.moe_utils import calc_scatter_index_stable


EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(False, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
torch.backends.cudnn.deterministic = False
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
    parser.add_argument("--drop_ratio", default=0.0, type=float, help="the token drop ratio")
    parser.add_argument(
        "--local_world_size", type=int, default=None, help="params use to simulate multiple nodes"
    )
    parser.add_argument("--insert_extra_barrier", action="store_true", default=False)
    parser.add_argument("--ep_teams", default=1, type=int, help="number of expert teams to split")
    return parser.parse_args()


def generate_random_exp_indices(token_num, total_num_experts, topk, drop_ratio):
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for tid in range(token_num):
        top_selected = random.sample(exp_list, topk)
        for i, _ in enumerate(top_selected):
            if random.uniform(0, 1) < drop_ratio:
                # current topk choice will be dropped
                top_selected[i] = total_num_experts
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


def perf_torch(input, exp_indices):
    assert exp_indices.size(1) == args.topk
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    # drop token logic :only need the splits information for the non-dropped tokenes
    if args.drop_ratio > 0:
        splits_gpu_cur_rank = splits_gpu_cur_rank[: args.G]
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
    count_before_drop = exp_indices.numel()
    count_after_drop = splits_cpu_cur_rank.sum()
    if args.drop_ratio > 0:
        print(f"Drop token enabled {count_before_drop} -> {count_after_drop}")
    if args.drop_ratio == 0:
        assert count_before_drop == count_after_drop
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
        if args.drop_ratio > 0:
            # the drop token logic here
            # fill the dropped ones with zero
            all2all_out_padded = torch.zeros(
                (count_before_drop, all2all_out.size(1)),
                device=all2all_out.device,
                dtype=all2all_out.dtype,
            )
            all2all_out_padded.data[:count_after_drop] = all2all_out
        else:
            all2all_out_padded = all2all_out
        gather_output = torch.zeros_like(all2all_out_padded)
        gather_output[new_index] = all2all_out_padded
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
            args.insert_extra_barrier,
        )
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return output, avg_time


def perf_flux_with_gpu_splits(input, exp_indices, num_recv_token):
    # print(f"exp_indices: {exp_indices.size()}")
    n_token_cur_ep_rank = exp_indices.size(0)
    input_len = torch.tensor([n_token_cur_ep_rank], dtype=torch.int32, device=input.device)
    ag_input_len = torch.zeros(EP_GROUP.size(), dtype=torch.int32, device=input.device)
    torch.distributed.all_gather_into_tensor(ag_input_len, input_len, group=EP_GROUP)
    ag_input_len_cpu = ag_input_len.cpu()
    ag_input_len_list = ag_input_len_cpu.tolist()
    padded_indices = torch.empty([args.M, args.topk], dtype=torch.int32, device=input.device)
    padded_indices[: exp_indices.size(0),] = exp_indices
    ag_padded_indices = [torch.empty_like(padded_indices) for _ in range(EP_GROUP.size())]
    # concat the exp_indices from all the rank
    torch.distributed.all_gather(ag_padded_indices, padded_indices, group=EP_GROUP)
    ag_indices = torch.concat(
        [t[: ag_input_len_list[i], :] for i, t in enumerate(ag_padded_indices)]
    )
    splits_gpu = torch.bincount(ag_indices.view(-1), minlength=args.G).to(torch.int32)
    output = torch.empty((num_recv_token, input.shape[1]), dtype=input.dtype, device=input.device)
    # simulate that the number of valid token is smaller than the size of buf
    input_buf = torch.empty(
        (int(input.shape[0] * 1.5), input.shape[1]), dtype=input.dtype, device=input.device
    )
    input_buf[: input.shape[0]].copy_(input)
    ag_scatter_idx = calc_scatter_index_stable(ag_indices)
    global op_gpu_splits
    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(args.iters):
        output = op_gpu_splits.forward_gpu(
            input_buf,
            ag_indices.view(-1),
            ag_scatter_idx.view(-1),
            splits_gpu,
            ag_input_len,
            output,
            args.sm_margin,
            args.insert_extra_barrier,
        )
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return output, avg_time


if __name__ == "__main__":
    args = parse_args()
    WORLD_GROUP = initialize_distributed()
    EP_GROUP = split_torch_process_group(WORLD_GROUP, args.ep_teams)
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if args.local_world_size is not None:
        LOCAL_WORLD_SIZE = args.local_world_size

    NVSHMEM_TEAM_WORLD = 0
    group_size = EP_GROUP.size()
    assert group_size >= LOCAL_WORLD_SIZE
    ep_team, pp_team = nvshmem_split_team_2d(group_size)
    mype_pp = flux._nvshmem_team_my_pe(pp_team)
    mype_ep = flux._nvshmem_team_my_pe(ep_team)
    npes_pp = flux._nvshmem_team_npes(pp_team)
    npes_ep = flux._nvshmem_team_npes(ep_team)
    mype = flux._nvshmem_team_translate_pe(
        pp_team, mype_pp, NVSHMEM_TEAM_WORLD
    )  # translate pp team to ep team
    mype2 = flux._nvshmem_team_translate_pe(
        ep_team, mype_ep, NVSHMEM_TEAM_WORLD
    )  # translate pp team to ep team
    torch.cuda.synchronize()
    torch.distributed.barrier(group=WORLD_GROUP)
    torch.cuda.synchronize()
    assert mype == RANK and mype2 == RANK, f"mype {mype} != mype_ep {RANK}"
    RANK_EP, WORLD_SIZE_EP = EP_GROUP.rank(), EP_GROUP.size()
    moe_capacity = 2.0
    op = flux.DisScatterBackward(
        args.G,
        args.M,
        args.N,
        args.topk,
        RANK_EP,
        1,
        WORLD_SIZE_EP,
        LOCAL_WORLD_SIZE,
        moe_capacity,
        ep_team,
    )
    # user need to ensure that the scale factor is large enough
    op_gpu_splits = flux.DisScatterBackward(
        args.G,
        args.M,
        args.N,
        args.topk,
        RANK_EP,
        1,
        WORLD_SIZE_EP,
        LOCAL_WORLD_SIZE,
        moe_capacity,
        ep_team,
    )

    torch.use_deterministic_algorithms(False)
    for rid in range(args.rounds):
        # random simulate token received from dataloader
        token_num = random.randint(args.M - 64, args.M)
        print(f"Received {token_num} tokens from dataloader @ rank:{RANK}")
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk, args.drop_ratio)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
        # handle drop token here, all the dropped tokens are routed the last non-exist expert

        # construct the input after the groupgemm
        if args.drop_ratio > 0:
            splits_gpu_cur_rank = splits_gpu_cur_rank[: args.G]
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
            flux_out_gpu_splits, flux_gpu_splits_time = perf_flux_with_gpu_splits(
                input, exp_indices, token_num
            )
        print(ref_out.size())
        print(flux_out.size())
        print(flux_out_gpu_splits.size())
        torch.cuda.synchronize()
        print(flux_out_gpu_splits)
        print(flux_out)
        print(ref_out)
        print(
            f"sum: torch:{torch.sum(ref_out)} flux:{torch.sum(flux_out)}, flux_gpu_splits:{torch.sum(flux_out_gpu_splits)}"
        )
        print(f"Flux: Time {flux_time} ms")
        print(f"Flux with gpu splts: Time {flux_gpu_splits_time} ms")
        print(f"Torch: Time {ref_time} ms")

        torch.distributed.barrier()
        flux.torch_allclose(flux_out, ref_out, 1e-2, 1e-2)
        flux.torch_allclose(flux_out_gpu_splits, flux_out, 0, 0)
