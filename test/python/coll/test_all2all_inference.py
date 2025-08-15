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
from flux import calc_scatter_index

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-N", type=int, default=6144)
    parser.add_argument("-G", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", default=1, type=int, help="perf iterations")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--sm_margin", default=16, type=int, help="sm margin")
    parser.add_argument(
        "--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys())
    )
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--with_scale", action="store_true", default=False)
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


def perf_torch(input, scale_tensor, exp_indices):
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    # calculate the scatter idx
    # scatter_idx_cur_rank = calc_scatter_index(exp_indices, splits_gpu_cur_rank)
    scatter_idx_cur_rank = calc_scatter_index_stable(exp_indices)
    # calculate the gather idx accordingly
    gather_idx_cur_rank, _ = calc_gather_index(scatter_idx_cur_rank, 0, token_num * args.topk)
    # use torch native scatter forward(will not be included in the e2e time measurement)
    scattered_input = torch.empty(
        input.size(0) * args.topk, input.size(1), dtype=input.dtype, device=input.device
    )
    scattered_scale_tensor = torch.empty(
        (scale_tensor.size(0) * args.topk), dtype=scale_tensor.dtype, device=scale_tensor.device
    )
    scattered_scale_tensor.copy_(torch.index_select(scale_tensor, dim=0, index=gather_idx_cur_rank))
    scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))
    # following are the all2all
    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    a2a_splits_cpu = a2a_splits.cpu()
    ep_size = EP_GROUP.size()
    a2a_dispatch_output = torch.empty(
        [a2a_splits_cpu.sum(), input.size(1)], dtype=input.dtype, device=input.device
    )
    a2a_dispatch_scale = torch.empty(
        [a2a_splits_cpu.sum()], dtype=scale_tensor.dtype, device=scale_tensor.device
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
        if args.with_scale:
            torch.distributed.all_to_all_single(
                output=a2a_dispatch_scale,
                input=scattered_scale_tensor,
                output_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
                input_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(-1).tolist(),
                group=EP_GROUP,
            )
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return a2a_dispatch_output, a2a_dispatch_scale, avg_time


def perf_flux(input, scale_tensor, exp_indices):
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    splits_cum_sum = torch.cumsum(splits_gpu_cur_rank, 0)
    splits_cum_sum_cpu = [0] + splits_cum_sum.cpu().tolist()
    splits_cum_sum = torch.tensor(splits_cum_sum_cpu).cuda().to(torch.int32)
    # calculate the scatter idx
    # scatter_idx_cur_rank = calc_scatter_index(exp_indices, splits_gpu_cur_rank)
    scatter_idx_cur_rank = calc_scatter_index_stable(exp_indices)
    # calculate the gather idx accordingly
    gather_idx_cur_rank, _ = calc_gather_index(scatter_idx_cur_rank, 0, token_num * args.topk)
    # use torch native scatter forward(will not be included in the e2e time measurement)
    scattered_input = torch.empty(
        input.size(0) * args.topk, input.size(1), dtype=input.dtype, device=input.device
    )
    scattered_scale_tensor = torch.empty(
        (scale_tensor.size(0) * args.topk), dtype=scale_tensor.dtype, device=scale_tensor.device
    )
    scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))
    scattered_scale_tensor.copy_(torch.index_select(scale_tensor, dim=0, index=gather_idx_cur_rank))
    # following are the all2all
    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    a2a_splits_cpu = a2a_splits.cpu()
    # the last op should directly write the output buffer into the comm buffer
    flux_input_tensors = op.get_input_buffer(scattered_input.size(), 2, args.with_scale)
    flux_input_tensors[0].copy_(scattered_input)
    print(flux_input_tensors[0])
    print(scattered_input)
    print(flux_input_tensors[0].size())
    print(scattered_input.size())
    assert torch.allclose(scattered_input, flux_input_tensors[0])
    if args.with_scale:
        flux_input_tensors[1].copy_(scattered_scale_tensor)
    ep_size = EP_GROUP.size()
    print("output splits ref", a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1))
    print("split_cum_sum:", splits_cum_sum)
    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(args.iters):
        flux_out = op.forward(
            [scattered_input.size(0), scattered_input.size(1)],
            splits_cum_sum,
            2,
            args.with_scale,
        )
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    data_vec = []
    scale_vec = []
    print("output_splits received", flux_out[0].cpu())
    output_splits = flux_out[0].cpu().reshape(ep_size, -1).sum(dim=-1)
    print("output_splits received", output_splits)
    for i in range(WORLD_SIZE):
        n_token_from_tgt_rank = output_splits[i]
        _start = i * args.M * args.topk
        data_vec.append(flux_out[1][_start : _start + n_token_from_tgt_rank])
        if args.with_scale:
            assert len(flux_out) == 3
            scale_vec.append(flux_out[2][_start : _start + n_token_from_tgt_rank])

    output = torch.concat(data_vec)
    scale_output = torch.concat(scale_vec) if args.with_scale else None
    return output, scale_output, avg_time


if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if args.local_world_size is not None:
        LOCAL_WORLD_SIZE = args.local_world_size
    RANK, WORLD_SIZE, NNODES = EP_GROUP.rank(), EP_GROUP.size(), flux.testing.NNODES()
    op = flux.All2AllInference(
        args.M * args.topk, args.N, RANK, args.G, WORLD_SIZE, LOCAL_WORLD_SIZE, 2
    )
    for rid in range(args.rounds):
        # random simulate token received from dataloader
        # token_num = args.M
        token_num = random.randint(args.M // 2, args.M)
        print(f"Received {token_num} tokens from dataloader @ rank:{RANK}")
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        input = torch.rand(token_num, args.N, dtype=torch.bfloat16).to("cuda")
        scale_tensor = torch.rand(token_num, dtype=torch.float32).to("cuda")
        with flux.group_profile(name="all2all_inference", group=EP_GROUP, do_prof=args.profile):
            ref_out, ref_scale, ref_time = perf_torch(input, scale_tensor, exp_indices)
            torch.cuda.synchronize()
            flux_out, flux_scale, flux_time = perf_flux(input, scale_tensor, exp_indices)
        print(ref_out.size())
        print(flux_out.size())
        torch.cuda.synchronize()
        print(flux_out)
        print(ref_out)
        print(f"sum: torch:{torch.sum(ref_out)} flux:{torch.sum(flux_out)}")
        print(f"Flux: Time {flux_time} ms")
        print(f"Torch: Time {ref_time} ms")

        torch.distributed.barrier()
        print(flux_out.size())
        print(ref_out.size())
        flux.torch_allclose(flux_out, ref_out, 1e-5, 1e-5)
        if args.with_scale:
            print("Scaled tensor:")
            print(flux_scale)
            print(ref_scale)
            flux.torch_allclose(flux_scale, ref_scale, 1e-5, 1e-5)
