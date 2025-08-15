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
import datetime
import os
import time
from contextlib import nullcontext
from functools import partial
from typing import Optional, List
import random

import numpy as np
import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import (
    DTYPE_MAP,
    initialize_distributed,
    split_torch_process_group,
    nvshmem_split_team_2d,
)

print = partial(print, flush=True)


class PerfResult:
    def __init__(
        self,
        name: str,
        a2a_output: torch.Tensor,
        total_ms: float,
    ) -> None:
        self.name = name
        self.a2a_output = a2a_output
        self.total_ms = total_ms

    def __repr__(self) -> str:
        return f"{self.name}: total {self.total_ms:.3f} ms"


def check_correctness(args, ep_team):

    num_iteration = args.iters
    dtype = DTYPE_MAP[args.dtype]
    max_split = args.M // WORLD_SIZE_EP
    flux_a2a_single = flux.All2AllSingle(
        EP_GROUP, max_split, args.N, args.local_world_size, dtype, ep_team
    )

    def _gen_inputs(splits, iter):
        input_splits = torch.tensor(
            list(np.random.randint(max(1, splits // 2), splits, size=(WORLD_SIZE_EP,))),
            dtype=torch.int32,
        ).cuda()
        output_splits = torch.zeros_like(input_splits)
        torch.distributed.all_to_all_single(output_splits, input_splits, group=EP_GROUP)
        input_shape = (input_splits.sum().cpu().item(), args.N)
        input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) * (RANK_EP + 1)
        input_splits_cpu = input_splits.cpu().tolist()
        output_splits_cpu = output_splits.cpu().tolist()
        return (input, input_splits, output_splits, input_splits_cpu, output_splits_cpu)

    def _torch_impl(input, input_splits, output_splits, input_splits_cpu, output_splits_cpu):
        output = torch.empty(
            (sum(output_splits_cpu), input.shape[1]), dtype=input.dtype, device=input.device
        )
        torch.distributed.all_to_all_single(
            output, input, output_splits_cpu, input_splits_cpu, group=EP_GROUP
        )
        return output

    def _flux_impl(input, input_splits, output_splits, input_splits_cpu, output_splits_cpu):
        output = torch.empty(
            (sum(output_splits_cpu), input.shape[1]), dtype=input.dtype, device=input.device
        )
        output = flux_a2a_single.forward(
            input,
            output,
            input_splits,
            output_splits,
            args.num_comm_sm,
        )
        return output

    all_inputs = [_gen_inputs(random.randint(1, max_split), iter) for iter in range(num_iteration)]
    torch_outputs = [_torch_impl(*inputs) for inputs in all_inputs]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    flux_outputs = [_flux_impl(*inputs) for inputs in all_inputs]

    torch.distributed.barrier()

    torch.cuda.synchronize()
    cnt = 0
    for idx, (flux_output, torch_output) in enumerate(zip(flux_outputs, torch_outputs)):
        flux_output = flux_output.reshape(torch_output.shape)

        atol, rtol = 0, 0
        try:
            flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
        except Exception as e:
            print(f"RANK {RANK_EP}, iter {idx}: ❌ flux check failed")
            raise e

    print("✅ flux check passed")

    EP_GROUP.barrier()
    torch.cuda.synchronize()


@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    warmup: int,
    iters: int,
):
    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    all2all_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    input_splits_cpu = input_splits.cpu().tolist()
    output_splits_cpu = output_splits.cpu().tolist()
    output = torch.empty(
        (sum(output_splits_cpu), input.shape[1]), dtype=input.dtype, device=input.device
    )
    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        torch.distributed.all_to_all_single(
            output, input, output_splits_cpu, input_splits_cpu, group=EP_GROUP
        )
        all2all_end_events[i].record()
        end_events[i].record()

    comm_times = []
    for i in range(total_iters):
        all2all_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{EP_GROUP.rank()}",
        a2a_output=output,
        total_ms=comm_time,
    )


@torch.no_grad()
def perf_flux(
    input: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    max_split: int,
    local_world_size: int,
    num_comm_sm: int,
    warmup: int,
    iters: int,
    ep_team: int,
):
    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    flux_a2a_single = flux.All2AllSingle(
        EP_GROUP, max_split, input.shape[1], local_world_size, input.dtype, ep_team
    )
    output_buf = torch.empty(
        (max_split * EP_GROUP.size(), input.size(1)), dtype=input.dtype, device=input.device
    )
    for i in range(total_iters):
        start_events[i].record()
        output_buf = flux_a2a_single.forward(
            input,
            output_buf,
            input_splits,
            output_splits,
            num_comm_sm,
        )
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    comm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times)

    comm_time_ms = comm_time / iters * 1000

    return PerfResult(
        name=f"flux  #{EP_GROUP.rank()}",
        a2a_output=output_buf,
        total_ms=comm_time_ms,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)

    parser.add_argument("--local_world_size", type=int, default=2, help="local world size")
    parser.add_argument("--num_comm_sm", type=int, default=8, help="num comm sm")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument(
        "--check",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="check correctness",
    )
    parser.add_argument("--ep_teams", default=1, type=int, help="number of expert teams")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    WORLD_GROUP = initialize_distributed()
    EP_GROUP = split_torch_process_group(WORLD_GROUP, args.ep_teams)
    ep_group_size = EP_GROUP.size()
    torch.use_deterministic_algorithms(False)
    RANK_EP, WORLD_SIZE_EP = EP_GROUP.rank(), EP_GROUP.size()
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)
    torch.cuda.synchronize()
    NVSHMEM_TEAM_WORLD = 0  # the global nvshmem world
    ep_team, pp_team = nvshmem_split_team_2d(ep_group_size)
    mype_pp = flux._nvshmem_team_my_pe(pp_team)
    mype_ep = flux._nvshmem_team_my_pe(ep_team)
    npes_pp = flux._nvshmem_team_npes(pp_team)
    npes_ep = flux._nvshmem_team_npes(ep_team)
    print(f"PP group: {mype_pp}/{npes_pp}, EP group: {mype_ep}/{npes_ep}")
    print(f"ep_team:{ep_team} pp_team:{pp_team}")
    mype = flux._nvshmem_team_translate_pe(
        pp_team, mype_pp, NVSHMEM_TEAM_WORLD
    )  # translate pp team to ep team
    assert mype == WORLD_GROUP.rank(), f"mype {mype} != mype_ep {WORLD_GROUP.rank()}"

    dtype = DTYPE_MAP[args.dtype]

    if args.check:
        check_correctness(args, ep_team)
        exit(0)

    max_split = args.M // WORLD_SIZE_EP
    input_splits = torch.tensor(
        list(np.random.randint(max(1, max_split - 32), max_split, size=(WORLD_SIZE_EP,))),
        dtype=torch.int32,
    ).cuda()
    output_splits = torch.zeros_like(input_splits)
    torch.distributed.all_to_all_single(output_splits, input_splits, group=EP_GROUP)
    print(f"RANK {RANK_EP}: input_splits = {input_splits}, output_splits = {output_splits}")

    input_shape = (input_splits.sum().cpu().item(), args.N)
    input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) * (RANK_EP + 1)

    torch.distributed.barrier()

    with flux.group_profile(name="all2all_single", group=EP_GROUP, do_prof=args.profile):
        perf_res_torch = perf_torch(
            input,
            input_splits,
            output_splits,
            args.warmup,
            args.iters,
        )

        perf_res_flux = perf_flux(
            input,
            input_splits,
            output_splits,
            max_split,
            args.local_world_size,
            args.num_comm_sm,
            args.warmup,
            args.iters,
            ep_team,
        )

    for i in range(EP_GROUP.size()):
        if i == EP_GROUP.rank():
            print(perf_res_torch)
            print(perf_res_flux)
        torch.distributed.barrier()

    torch_output = perf_res_torch.a2a_output
    flux_output = perf_res_flux.a2a_output[: torch_output.shape[0]]  # slice valid data
    torch.distributed.barrier()

    atol, rtol = 0.0, 0.0
    try:
        flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
    except Exception as e:
        print("❌ flux check failed")
        raise e
    else:
        print("✅ flux check passed")

    EP_GROUP.barrier()
    torch.cuda.synchronize()
