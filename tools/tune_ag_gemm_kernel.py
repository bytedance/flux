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
import dataclasses
import datetime
import itertools
import os
import time
from functools import partial
from typing import List

import numpy as np
import torch

import flux
import flux.testing

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE
torch.cuda.set_device(LOCAL_RANK)

os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
np.random.seed(3 + RANK)

torch.distributed.init_process_group(
    backend="nccl", world_size=WORLD_SIZE, rank=RANK, timeout=datetime.timedelta(seconds=1800)
)
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
print = partial(print, flush=True)


@dataclasses.dataclass
class TuningConfig:
    M: int
    N: int
    K: int
    transpose_weight: bool
    dtype: str
    has_bias: bool


def gen_tuning_space(dtype, check: bool):
    space: List[TuningConfig] = []
    ## Training shapes
    # space_M = [4096]
    # space_N = [10240, 24576, 8192, 57344, 28672]
    # space_K = [8192]
    ## EP Serving
    # space_M = [512, 256, 128, 64]
    space_M = [512, 256, 128, 64] if dtype == torch.int8 else [4096, 64, 256, 512, 1024, 2048, 8192]
    space_N = [10240] if dtype == torch.int8 else [49152, 16384]
    space_K = [8192 if dtype == torch.int8 else 12288]
    space_transpose_weight = [False] if dtype == torch.int8 else [True, False]
    space_dtype = [dtype] if dtype == torch.int8 else [torch.bfloat16, torch.float16]
    space_has_bias = [True] if dtype == torch.int8 else [False, True]

    ## Check only config for CI to save time
    if check:
        space_M = [space_M[0]]
        space_N = [space_N[0]]
        space_K = [space_K[0]]
        space_transpose_weight = [space_transpose_weight[0]]
        space_dtype = [space_dtype[0]]
        space_has_bias = [space_has_bias[0]]

    for M, N, K, transpose_weight, dtype, has_bias in itertools.product(
        space_M,
        space_N,
        space_K,
        space_transpose_weight,
        space_dtype,
        space_has_bias,
    ):
        config = TuningConfig(
            M=M,
            N=N,
            K=K,
            transpose_weight=transpose_weight,
            dtype=dtype,
            has_bias=has_bias,
        )
        space.append(config)
    return space


def get_torch_output(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    is_s8_dequant: bool,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()

    torch.distributed.barrier()
    full_input = torch.zeros(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    full_input_scale = (
        torch.zeros(
            (M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device(), requires_grad=False
        )
        if is_s8_dequant
        else None
    )
    if is_s8_dequant:
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
    torch.distributed.barrier()
    if is_s8_dequant:
        accum = flux.testing.matmul_int8(full_input, weight.t()).to(torch.float32)
        output = full_input_scale * weight_scale * accum
    else:
        output = 1.0 * torch.matmul(full_input, weight.t())

    if is_s8_dequant:
        output = output.to(torch.bfloat16)
    if bias is not None:
        output += bias
    torch.distributed.barrier()
    return output


def run_flux_profiling(
    prof_ctx: flux.ProfilingContext,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    is_s8_dequant: bool,
    config: TuningConfig,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()
    K = input.size(1)

    if config.transpose_weight:
        w = weight.t().contiguous()
        N = w.size(1)
    else:
        w = weight
        N = w.size(0)

    output_dtype = torch.bfloat16 if is_s8_dequant else input.dtype
    ag_gemm_kernel = flux.AGKernel(
        TP_GROUP,
        NNODES,
        M,
        N,
        K,
        input.dtype,
        output_dtype=output_dtype,
    )

    output = ag_gemm_kernel.profiling(
        input,
        w,
        bias=bias,
        output=None,
        input_scale=input_scale,
        weight_scale=weight_scale,
        output_scale=None,
        fast_accum=False,
        transpose_weight=config.transpose_weight,
        gathered_input=None,
        prof_ctx=prof_ctx,
    )
    return output


def tune_one_config(prof_ctx: flux.ProfilingContext, config: TuningConfig):
    assert config.M % TP_GROUP.size() == 0
    assert config.N % TP_GROUP.size() == 0
    local_M = config.M // TP_GROUP.size()
    local_N = config.N // TP_GROUP.size()

    is_s8_dequant = config.dtype == torch.int8

    # input: [M, K], weight: [N, K]
    input = None
    weight = None
    if is_s8_dequant:
        input = torch.randint(-120, 120, (local_M, config.K), dtype=torch.int8).cuda()
        weight = torch.randint(-120, 120, (local_N, config.K), dtype=torch.int8).cuda()
    else:
        input = (
            torch.rand((local_M, config.K), dtype=config.dtype).cuda()
            / 100
            * ((TP_GROUP.rank() + 1))
        )
        weight = (
            torch.rand((local_N, config.K), dtype=config.dtype).cuda()
            / 100
            * ((TP_GROUP.rank() + 1))
        )

    input_scale = None
    weight_scale = None

    if is_s8_dequant:
        input_scale = torch.rand((local_M, 1), dtype=torch.float32).cuda()
        weight_scale = torch.rand((1, local_N), dtype=torch.float32).cuda()

    bias = None
    if config.has_bias:
        if is_s8_dequant:
            bias = torch.rand((1, local_N), dtype=torch.bfloat16, device=input.device)
        else:
            bias = torch.rand((config.M, local_N), dtype=input.dtype, device=input.device)

    torch_output = get_torch_output(input, weight, bias, input_scale, weight_scale, is_s8_dequant)
    torch.distributed.barrier()
    flux_output = run_flux_profiling(
        prof_ctx, input, weight, bias, input_scale, weight_scale, is_s8_dequant, config
    )
    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    if config.dtype == torch.bfloat16:
        atol, rtol = 0.02, 0.02
    else:
        atol, rtol = 0.01, 0.01

    for rank in range(WORLD_SIZE):
        if rank == RANK:
            flux.torch_allclose(flux_output.cpu(), torch_output.cpu(), atol=atol, rtol=rtol)
        torch.distributed.barrier()

    time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="", type=str, help="Directory to store generated files"
    )
    parser.add_argument("--dtype", default="bf16", type=str, help="data type")
    parser.add_argument(
        "--check",
        default=False,
        action="store_true",
        help="Check several sizes for functional test",
    )
    return parser.parse_args()


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "s8": torch.int8,
}

if __name__ == "__main__":
    args = parse_args()
    if args.output_dir and not os.path.isdir(args.output_dir):
        raise Exception(f"{args.output_dir} not exist")

    torch.cuda.set_device(LOCAL_RANK)
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()

    arch: int = flux.get_arch()
    name: str = f"config_ag_gemm_kernel_sm{arch}_tp{WORLD_SIZE}_nnodes{NNODES}"
    prof_ctx = flux.ProfilingContext(name)
    dtype = DTYPE_MAP[args.dtype]
    config_space = gen_tuning_space(dtype, args.check)
    for i, config in enumerate(config_space):
        if TP_GROUP.rank() == 0:
            print(f"==== #{i + 1}/{len(config_space)} Tuning for {config}")
        tune_one_config(prof_ctx=prof_ctx, config=config)
        if TP_GROUP.rank() == 0:
            print(prof_ctx.get_latest_prof_result())

    if TP_GROUP.rank() == 0:
        if os.path.isdir(args.output_dir):
            code_path = os.path.join(args.output_dir, f"{name}.cu")
            result_path = os.path.join(args.output_dir, f"{name}.prof.log")

            with open(code_path, "w") as fout:
                print(prof_ctx.get_code(), file=fout)

            with open(result_path, "w") as fout:
                for record in prof_ctx.get_all_prof_results():
                    print(record, file=fout)
        else:
            print("Generated Code:")
            print(prof_ctx.get_code())
            print()

            print("Profiling Results:")
            for record in prof_ctx.get_all_prof_results():
                print(record)
