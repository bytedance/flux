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
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-G", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--iters", default=1, type=int, help="perf iterations")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--sm_margin", default=16, type=int, help="sm margin")
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--drop_ratio", default=0.0, type=float, help="the token drop ratio")
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
    return torch.Tensor(exp_indices).to(torch.int32)


def all_gather_v_forward(inputs, all_rank_shapes, ep_group):
    output = torch.empty(
        [all_rank_shapes.sum(), inputs.shape[-1]], device=inputs.device, dtype=inputs.dtype
    )
    output_list = list(torch.split(output, all_rank_shapes.tolist(), dim=0))
    torch.distributed.all_gather(output_list, inputs, group=ep_group)
    return output


def perf_torch(tokens_per_rank_cpu, topk_indices, topk_vals):
    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(args.iters):
        gathered_indices = all_gather_v_forward(topk_indices, tokens_per_rank_cpu, EP_GROUP)
        gathered_values = all_gather_v_forward(topk_vals, tokens_per_rank_cpu, EP_GROUP)
        # a2a_splits = torch.empty_like(splits)
        # torch.distributed.all_to_all_single(a2a_splits, splits, group=EP_GROUP)

    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return gathered_indices, gathered_values, avg_time


def perf_flux(ag_input_len, topk_indices, topk_vals):
    global op
    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(args.iters):
        ag_idx, ag_val = op.pre_comm_index(ag_input_len, topk_indices, topk_vals, args.sm_margin)
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return ag_idx, ag_val, avg_time


def perf_flux_with_gpu_splits(ag_input_len, topk_indices, topk_vals):
    global op_gpu_splits
    ag_topk_indices = torch.empty(
        [ag_input_len.sum(), topk_indices.shape[-1]],
        device=topk_indices.device,
        dtype=topk_indices.dtype,
    )
    ag_topk_values = torch.empty(
        [ag_input_len.sum(), topk_vals.shape[-1]], device=topk_vals.device, dtype=topk_vals.dtype
    )
    ag_input_len_gpu = ag_input_len.cuda()
    torch.cuda.synchronize()
    time_start = time.time()
    for _ in range(args.iters):
        ag_idx, ag_val = op_gpu_splits.pre_comm_index_gpu(
            ag_input_len_gpu,
            topk_indices,
            topk_vals,
            ag_topk_indices,
            ag_topk_values,
            args.sm_margin,
        )
    torch.cuda.synchronize()
    time_end = time.time()
    avg_time = (time_end - time_start) / args.iters * 1000
    return ag_idx, ag_val, avg_time


if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()
    torch.use_deterministic_algorithms(False)
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    RANK, WORLD_SIZE, NNODES = EP_GROUP.rank(), EP_GROUP.size(), flux.testing.NNODES()
    op = flux.DisScatterForward(
        args.G, args.M, 8192, args.topk, RANK, 1, WORLD_SIZE, LOCAL_WORLD_SIZE
    )
    op_gpu_splits = flux.DisScatterForward(
        args.G, args.M, 8192, args.topk, RANK, 1, WORLD_SIZE, LOCAL_WORLD_SIZE
    )
    for rid in range(args.rounds):
        # random simulate token received from dataloader
        token_num = random.randint(args.M // 2, args.M)
        # token_num = args.M
        print(f"Received {token_num} tokens from dataloader @ rank:{RANK}")
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk, args.drop_ratio)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        exp_topk_val = torch.rand((token_num, args.topk), dtype=torch.float).cuda()
        input_len = torch.tensor([token_num], dtype=torch.int32, device=exp_indices.device)
        ag_input_len = torch.zeros(EP_GROUP.size(), dtype=torch.int32, device=exp_indices.device)
        torch.distributed.all_gather_into_tensor(ag_input_len, input_len, group=EP_GROUP)
        ag_input_len_cpu = ag_input_len.cpu()
        splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
        with flux.group_profile(name="all2all_index_comm", group=EP_GROUP, do_prof=args.profile):
            ag_topk_idx_ref, ag_topk_val_ref, ref_time = perf_torch(
                ag_input_len_cpu, exp_indices, exp_topk_val
            )
            torch.cuda.synchronize()
            ag_topk_idx, ag_topk_val, flux_time = perf_flux(
                ag_input_len_cpu, exp_indices, exp_topk_val
            )
            torch.cuda.synchronize()
            ag_topk_idx_with_gpu_splits, ag_topk_val_with_gpu_splits, flux_gpu_splits_time = (
                perf_flux_with_gpu_splits(ag_input_len_cpu, exp_indices, exp_topk_val)
            )
            torch.cuda.synchronize()

        print(f"Flux gpu spilits: Time {flux_gpu_splits_time} ms")
        print(f"Flux: Time {flux_time} ms")
        print(f"Torch: Time {ref_time} ms")
        torch.distributed.barrier()
        flux.torch_allclose(ag_topk_idx, ag_topk_idx_ref, 1e-3, 1e-3)
        flux.torch_allclose(ag_topk_val, ag_topk_val_ref, 1e-6, 1e-6)
        flux.torch_allclose(ag_topk_val, ag_topk_val_with_gpu_splits, 0, 0)
        flux.torch_allclose(ag_topk_idx, ag_topk_idx_with_gpu_splits, 0, 0)

        diff_sum = torch.sum(ag_topk_idx - ag_topk_idx_ref) + torch.sum(
            ag_topk_val - ag_topk_val_ref
        )
        print(f"dtype: {ag_topk_idx.dtype} {ag_topk_val.dtype}")
        print("diff_sum ", diff_sum)
        assert torch.allclose(ag_topk_idx, ag_topk_idx_ref)
        assert torch.allclose(ag_topk_val, ag_topk_val_ref)
