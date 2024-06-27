################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
from functools import partial
import logging
import sys
import torch
import numpy as np
import torch.distributed
import flux
from utils import TP_GROUP, init_seed, RANK, WORLD_SIZE, NNODES, LOCAL_RANK, LOCAL_WORLD_SIZE


print = partial(print, flush=True)


@torch.no_grad()
def run_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
):
    # All gather input tensors from all gpus
    full_input = torch.empty(
        (input.size(0) * LOCAL_WORLD_SIZE, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
    output = torch.matmul(full_input, weight.t())
    if bias is not None:
        output += bias

    return output, full_input


@torch.no_grad()
def run_flux_with_op(
    ag_op: flux.AGKernel,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    local_copy: bool = False,
):
    ag_op.reset_signals()
    if local_copy:
        ag_op.copy_local(input)
    output = ag_op.forward(input, weight, bias=bias)
    gathered = ag_op.gather()
    return output, gathered


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument(
        "--ring_mode",
        default=-1,
        type=int,
        help="ring mode. -1 for auto detect. 0 for false, 1 for true",
    )
    return parser.parse_args()


def run_test_with_args(
    M_max,
    local_N,
    K,
    dtype,
    transpose_weight: bool,
    local_copy: bool,
    has_bias: bool,
    ring_mode: flux.AgRingMode,
):
    def _run(local_m):
        # input: [M_max, K], weight: [N, K]
        input = (
            (-2 * torch.rand((local_m, K), dtype=dtype).cuda() + 1) / 100 * (TP_GROUP.rank() + 1)
        )
        weight = (
            (-2 * torch.rand((local_N, K), dtype=dtype).cuda() + 1) / 100 * (TP_GROUP.rank() + 1)
        )
        bias = (
            torch.rand((local_m * LOCAL_WORLD_SIZE, local_N), dtype=dtype).cuda()
            / 10
            * (TP_GROUP.rank() + 1)
            if has_bias
            else None
        )
        w = weight.t().contiguous() if transpose_weight else weight.contiguous()

        gt_result, gt_all_input = run_torch(input, weight, bias)
        flux_result, flux_all_input = run_flux_with_op(
            ag_op,
            input,
            w,
            bias,
            local_copy,
        )

        torch.distributed.barrier()
        flux.bitwise_check(gt_all_input, flux_all_input)
        flux.torch_allclose(gt_result, flux_result, atol=1e-02, rtol=1e-02)

    ag_op = flux.AGKernel(
        TP_GROUP,
        NNODES,
        M_max,
        local_N,
        K,
        dtype,
        transpose_weight=transpose_weight,
        local_copy=local_copy,
        ring_mode=ring_mode,
    )

    torch.distributed.barrier(TP_GROUP)
    local_m_max = M_max // LOCAL_WORLD_SIZE
    try:
        _run(local_m_max)
        print(f"run with local_m={local_m_max} done")
    except:
        logging.exception(f"run with local_m={local_m_max} failed")
    try:
        _run(local_m_max // 2)
        print(f"run with local_m={local_m_max // 2} done")
    except:
        logging.exception(f"run with local_m={local_m_max//2} failed")


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}

if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    init_seed()
    args = parse_args()

    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()

    assert args.M % TP_GROUP.size() == 0
    assert args.N % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_N = args.N // TP_GROUP.size()

    for dtype in [torch.float16, torch.bfloat16]:
        for local_copy in [True, False]:
            for transpose_weight in [True, False]:
                for has_bias in [True, False]:
                    print(
                        f"dtype={dtype}, local_copy={local_copy}, transpose_weight={transpose_weight}, has_bias={has_bias}"
                    )
                    run_test_with_args(
                        args.M,
                        local_N,
                        args.K,
                        dtype,
                        transpose_weight,
                        local_copy,
                        has_bias,
                        flux.AgRingMode(args.ring_mode),
                    )
