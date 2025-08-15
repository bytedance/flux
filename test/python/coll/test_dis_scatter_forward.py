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
import random
import time

import flux
import flux.testing
import numpy as np
import torch
import torch.distributed as dist
from flux.testing import (
    DTYPE_MAP,
    initialize_distributed,
    split_torch_process_group,
    nvshmem_split_team_2d,
)
from flux.testing.moe_utils import calc_gather_index, calc_scatter_index_stable

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
    parser.add_argument("--drop_ratio", default=0.0, type=float, help="the token drop ratio")
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument(
        "--local_world_size", type=int, default=None, help="params use to simulate multiple nodes"
    )
    parser.add_argument(
        "--copy_to_local",
        action="store_true",
        default=False,
        help="flag to control whether to copy the output from the communication buffer to a local buffer",
    )
    parser.add_argument("--ep_teams", default=1, type=int, help="number of expert teams")
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
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    # drop token logic :only need the splits information for the non-dropped tokenes
    splits_gpu_cur_rank = splits_gpu_cur_rank[: args.G]
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    count_after_drop = torch.sum(splits_cpu_cur_rank)
    count_origin = exp_indices.numel()
    print(
        f"Drop token : {count_origin} -> {count_after_drop}",
        torch.sum(splits_cpu_cur_rank),
        exp_indices.numel(),
        splits_cpu_cur_rank.size(),
    )
    # calculate the scatter idx
    # scatter_idx_cur_rank = calc_scatter_index(exp_indices, splits_gpu_cur_rank)
    scatter_idx_cur_rank = calc_scatter_index_stable(exp_indices)
    # calculate the gather idx accordingly
    gather_idx_cur_rank, _ = calc_gather_index(scatter_idx_cur_rank, 0, token_num * args.topk)
    # use torch native scatter forward(will not be included in the e2e time measurement)
    assert count_origin == input.size(0) * args.topk
    scattered_input = torch.empty(
        input.size(0) * args.topk, input.size(1), dtype=input.dtype, device=input.device
    )
    scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))
    # drop token logic: drop the token here
    scattered_input = scattered_input[:count_after_drop]
    # following are the all2all
    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    a2a_splits_cpu = a2a_splits.cpu()
    ep_size = EP_GROUP.size()
    a2a_dispatch_output = torch.empty(
        [a2a_splits_cpu.sum(), input.size(1)], dtype=input.dtype, device=input.device
    )
    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(args.iters):
        torch.distributed.all_to_all_single(
            output=a2a_dispatch_output,
            input=scattered_input,
            output_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
            input_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(-1).tolist(),
            group=EP_GROUP,
        )
        num_experts_per_rank = args.G // ep_size
        assert args.G % ep_size == 0
        a2a_expert_input_list = torch.split(a2a_dispatch_output, a2a_splits_cpu.tolist())
        permute_a2a_expert_input_list = list()
        for idx in range(num_experts_per_rank):
            for idy in range(ep_size):
                permute_a2a_expert_input_list.append(
                    a2a_expert_input_list[idy * num_experts_per_rank + idx]
                )
        permute_a2a_expert_input = torch.cat(permute_a2a_expert_input_list, dim=0)
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return permute_a2a_expert_input, avg_time


def perf_flux(input, exp_indices):
    input_len = torch.tensor([input.size(0)], dtype=torch.int32, device=input.device)
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
    print("splits_cpu: ", splits_cpu)
    print("splits_size", splits_cpu.size())
    # ag_scatter_idx = calc_scatter_index(ag_indices, splits_gpu)
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
            args.copy_to_local,
        )
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return output, avg_time


def perf_flux_with_gpu_splits(input, exp_indices, num_output_token):
    input_len = torch.tensor([input.size(0)], dtype=torch.int32, device=input.device)
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
    # ag_scatter_idx = calc_scatter_index(ag_indices, splits_gpu)
    ag_scatter_idx = calc_scatter_index_stable(ag_indices)
    output_buf = torch.empty(
        (int(num_output_token * 1.5), input.shape[1]), dtype=input.dtype, device=input.device
    )

    global op_gpu_splits
    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(args.iters):
        output = op_gpu_splits.forward_gpu(
            input,
            ag_indices.view(-1),
            ag_scatter_idx.view(-1),
            splits_gpu,
            ag_input_len,
            output_buf,
            args.sm_margin,
            args.copy_to_local,
        )
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return output, avg_time


# about the world_size and ranks
# about --local_world_size: this is used only in GLOBAL team, with which to check if nvshmem_ptr returns nullptr or not.
# it's best that a2a use the contiguous rank, and pp use the strided ranks.

if __name__ == "__main__":
    args = parse_args()
    WORLD_GROUP = initialize_distributed()
    EP_GROUP = split_torch_process_group(WORLD_GROUP, args.ep_teams)

    NVSHMEM_TEAM_WORLD = 0
    group_size = EP_GROUP.size()
    assert group_size >= args.local_world_size
    ep_team, pp_team = nvshmem_split_team_2d(group_size)
    mype_pp = flux._nvshmem_team_my_pe(pp_team)
    mype_ep = flux._nvshmem_team_my_pe(ep_team)
    npes_pp = flux._nvshmem_team_npes(pp_team)
    npes_ep = flux._nvshmem_team_npes(ep_team)
    print(f"PP group: {mype_pp}/{npes_pp}, EP group: {mype_ep}/{npes_ep}")
    print(f"ep_team:{ep_team} pp_team:{pp_team}")
    mype = flux._nvshmem_team_translate_pe(
        pp_team, mype_pp, NVSHMEM_TEAM_WORLD
    )  # translate pp team to ep team
    assert mype == RANK, f"mype {mype} != mype_ep {RANK}"

    torch.cuda.synchronize()
    torch.distributed.barrier(group=WORLD_GROUP)
    torch.cuda.synchronize()

    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    torch.use_deterministic_algorithms(False)
    if args.local_world_size is not None:
        LOCAL_WORLD_SIZE = args.local_world_size
    RANK_EP, WORLD_SIZE_EP = EP_GROUP.rank(), EP_GROUP.size()
    comm_buffer_duplicate = 1
    op = flux.DisScatterForward(
        args.G,
        args.M,
        args.N,
        args.topk,
        RANK_EP,
        1,
        WORLD_SIZE_EP,
        LOCAL_WORLD_SIZE,
        1,
        comm_buffer_duplicate,
        ep_team,
    )

    # TODO: check that the scale factor is large enough
    scale_factor = 2.0
    op_gpu_splits = flux.DisScatterForward(
        args.G,
        args.M,
        args.N,
        args.topk,
        RANK_EP,
        1,
        WORLD_SIZE_EP,
        LOCAL_WORLD_SIZE,
        scale_factor,
        comm_buffer_duplicate,
        ep_team,
    )

    for rid in range(args.rounds):
        # random simulate token received from dataloader
        token_num = random.randint(args.M - 64, args.M)
        print(f"Received {token_num} tokens from dataloader @ rank:{RANK_EP}")
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk, args.drop_ratio)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        token_drop_count = (exp_indices == (args.G + 1)).sum()
        print("Drop Token count: ", token_drop_count)
        exp_indices = exp_indices.to("cuda")
        input = torch.rand(token_num, args.N, dtype=torch.bfloat16).to("cuda")
        with flux.group_profile(name="all2all_dispatch", group=EP_GROUP, do_prof=args.profile):
            ref_out, ref_time = perf_torch(input, exp_indices)
            torch.cuda.synchronize()
            flux_out, flux_time = perf_flux(input, exp_indices)
            flux_gpu_splits_out, flux_gpu_splits_time = perf_flux_with_gpu_splits(
                input, exp_indices, flux_out.shape[0]
            )
        flux_gpu_splits_out = flux_gpu_splits_out[: flux_out.shape[0]]
        print(flux_gpu_splits_out.size())
        print(ref_out.size())
        print(flux_out.size())
        torch.cuda.synchronize()
        print(flux_gpu_splits_out)
        print(flux_out)
        print(ref_out)
        print(
            f"sum: torch:{torch.sum(ref_out)} flux:{torch.sum(flux_out)}, flux_gpu_splits:{torch.sum(flux_gpu_splits_out)}"
        )
        print(f"Flux: Time {flux_time} ms")
        print(f"Flux gpu splits: Time {flux_gpu_splits_time} ms")
        print(f"Torch: Time {ref_time} ms")

        torch.distributed.barrier()
        flux.torch_allclose(flux_out, ref_out, 1e-5, 1e-5)
        flux.torch_allclose(flux_gpu_splits_out, flux_out, 0, 0)
