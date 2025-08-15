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
    bs: int
    nh: int
    seq: int
    hd: int
    out_features: int
    transpose_weight: bool
    dtype: str
    has_bias: bool


def gen_tuning_space(dtype, check: bool):
    space: List[TuningConfig] = []
    ## SP shapes
    space_bs = [1]
    space_nh = [64]
    space_seq = [8192 * 2]
    space_hd = [128]
    space_out_features = [4096]
    space_transpose_weight = [False]
    space_dtype = [dtype]
    space_has_bias = [False]

    ## Check only config for CI to save time
    if check:
        space_bs = space_bs[:1]
        space_nh = space_nh[:1]
        space_seq = space_seq[:1]
        space_hd = space_hd[:1]
        space_out_features = space_out_features[:1]
        space_transpose_weight = space_transpose_weight[:1]
        space_dtype = space_dtype[:1]
        space_has_bias = space_has_bias[:1]

    for bs, nh, seq, hd, out_features, transpose_weight, dtype, has_bias in itertools.product(
        space_bs,
        space_nh,
        space_seq,
        space_hd,
        space_out_features,
        space_transpose_weight,
        space_dtype,
        space_has_bias,
    ):
        config = TuningConfig(
            bs=bs,
            nh=nh,
            seq=seq,
            hd=hd,
            out_features=out_features,
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
    a2a_only: bool,
):
    if not a2a_only:
        bs, local_nh, seq_len, hd = input.shape
    else:
        bs, seq_len, local_nh, hd = input.shape
    local_seq_len = seq_len // TP_GROUP.size()
    hidden_dim = local_nh * hd * TP_GROUP.size()

    torch.distributed.barrier()

    # All to all input tensors from all gpus
    input_after_a2a = torch.zeros(
        (seq_len, bs, local_nh, hd),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    def all_to_all_transpose():
        if not a2a_only:
            input_before_a2a = input.permute(2, 0, 1, 3).contiguous()
        else:
            input_before_a2a = input.permute(1, 0, 2, 3).contiguous()  # [seq_len, bs, local_nh, hd]
        torch.distributed.all_to_all_single(input_after_a2a, input_before_a2a, group=TP_GROUP)
        gemm_input = (
            input_after_a2a.reshape(TP_GROUP.size(), local_seq_len, bs, local_nh, hd)
            .permute(2, 1, 0, 3, 4)
            .reshape(bs, local_seq_len, hidden_dim)
        )
        return gemm_input

    gemm_input = all_to_all_transpose()
    torch.distributed.barrier()
    output = torch.matmul(gemm_input, weight.t())

    if bias is not None:
        output += bias
    torch.distributed.barrier()
    return output


def run_flux_profiling(
    prof_ctx: flux.ProfilingContext,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    config: TuningConfig,
    a2a_only: bool,
):
    input_dtype = input.dtype
    if not a2a_only:
        bs, local_nh, seq_len, hd = input.shape
    else:
        bs, seq_len, local_nh, hd = input.shape
    nh = local_nh * TP_GROUP.size()
    local_seq_len = seq_len // TP_GROUP.size()
    hidden_dim = local_nh * hd * TP_GROUP.size()
    gemm_m = bs * local_seq_len
    gemm_k = hidden_dim
    if config.transpose_weight:
        w = weight.t().contiguous()
        gemm_n = w.shape[1]
    else:
        w = weight
        gemm_n = w.shape[0]

    output_dtype = input.dtype
    option = flux.AllToAllOption()
    option.fuse_sync = True
    a2a_gemm_kernel = flux.AllToAllTransposeGemm(
        TP_GROUP,
        NNODES,
        bs,
        nh,
        seq_len,
        hd,
        input_dtype,
        output_dtype=output_dtype,
        a2a_only=a2a_only,
    )

    output = a2a_gemm_kernel.profiling(
        input,
        w,
        bias=bias,
        output=None,
        input_scale=None,
        weight_scale=None,
        output_scale=None,
        fast_accum=False,
        transpose_weight=config.transpose_weight,
        all_to_all_option=option,
        num_comm_sm=16,
        prof_ctx=prof_ctx,
    )
    return output


def tune_one_config(prof_ctx: flux.ProfilingContext, config: TuningConfig, a2a_only: bool):
    assert config.seq % TP_GROUP.size() == 0
    assert config.nh % TP_GROUP.size() == 0
    local_nh = config.nh // TP_GROUP.size()
    local_seq = config.seq // TP_GROUP.size()

    hidden_dim = config.nh * config.hd
    if not a2a_only:
        input_shape = [config.bs, local_nh, config.seq, config.hd]
    else:
        input_shape = [config.bs, config.seq, local_nh, config.hd]
    weight_shape = [config.out_features, hidden_dim]

    input = (
        (-2 * torch.rand(input_shape, dtype=config.dtype).cuda() + 1) / 100 * (TP_GROUP.rank() + 1)
    )
    weight = (
        (-2 * torch.rand(weight_shape, dtype=config.dtype).cuda() + 1) / 100 * (TP_GROUP.rank() + 1)
    )

    input_scale = None
    weight_scale = None

    bias = None
    gemm_m = config.bs * local_seq
    bias_shape = [gemm_m, config.out_features]
    if config.has_bias:
        bias = torch.rand(bias_shape, dtype=dtype).cuda() / 10 * (TP_GROUP.rank() + 1)

    torch_output = get_torch_output(input, weight, bias, a2a_only)
    torch.distributed.barrier()
    flux_output = run_flux_profiling(prof_ctx, input, weight, bias, config, a2a_only)
    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    if config.dtype == torch.bfloat16:
        atol, rtol = 0.02, 0.02
    else:
        atol, rtol = 0.01, 0.01

    for rank in range(WORLD_SIZE):
        if rank == RANK:
            flux_output = flux_output.reshape(torch_output.shape)
            print(torch.max(torch.abs(flux_output - torch_output)))
            flux.torch_allclose(flux_output.cpu(), torch_output.cpu(), atol=atol, rtol=rtol)
        torch.distributed.barrier()

    time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="", type=str, help="Directory to store generated files"
    )
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--check",
        default=False,
        action="store_true",
        help="Check several sizes for functional test",
    )
    parser.add_argument(
        "--a2a_only", default=False, action="store_true", help="whether have transpose"
    )
    return parser.parse_args()


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
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
    name: str = f"config_a2a_gemm_kernel_sm{arch}_tp{WORLD_SIZE}_nnodes{NNODES}"
    prof_ctx = flux.ProfilingContext(name)
    dtype = DTYPE_MAP[args.dtype]
    config_space = gen_tuning_space(dtype, args.check)
    for i, config in enumerate(config_space):
        if TP_GROUP.rank() == 0:
            print(f"==== #{i + 1}/{len(config_space)} Tuning for {config}")
        tune_one_config(prof_ctx=prof_ctx, config=config, a2a_only=args.a2a_only)
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
