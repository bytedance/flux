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
    seq: int
    hidden_dim: int
    head_dim: int
    out_features: int
    transpose_weight: bool
    dtype: str
    has_bias: bool
    num_comm_sm: int


def gen_tuning_space(dtype, check: bool):
    space: List[TuningConfig] = []
    ## SP shapes
    # space_M = [2048]
    # space_N = [8192]
    # space_K = [4096]
    space_bs = [2]
    space_seq = [8192]
    space_hidden_dim = [4096]
    space_head_dim = [128]
    space_out_features = [10240]
    space_transpose_weight = [False]
    space_dtype = [dtype]
    space_has_bias = [False]
    space_num_comm_sm = [16, 20, 24]

    ## Check only config for CI to save time
    if check:
        space_bs = space_bs[:1]
        space_seq = space_seq[:1]
        space_hidden_dim = space_hidden_dim[:1]
        space_head_dim = space_head_dim[:1]
        space_out_features = space_out_features[:1]
        space_transpose_weight = space_transpose_weight[:1]
        space_dtype = space_dtype[:1]
        space_has_bias = space_has_bias[:1]
        space_num_comm_sm = space_num_comm_sm[:1]

    for (
        bs,
        seq,
        hidden_dim,
        head_dim,
        out_features,
        transpose_weight,
        dtype,
        has_bias,
        num_comm_sm,
    ) in itertools.product(
        space_bs,
        space_seq,
        space_hidden_dim,
        space_head_dim,
        space_out_features,
        space_transpose_weight,
        space_dtype,
        space_has_bias,
        space_num_comm_sm,
    ):
        config = TuningConfig(
            bs=bs,
            seq=seq,
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            out_features=out_features,
            transpose_weight=transpose_weight,
            dtype=dtype,
            has_bias=has_bias,
            num_comm_sm=num_comm_sm,
        )
        space.append(config)
    return space


def _torch_all_to_all_transpose(input, bs, seq_len, nh, head_dim):
    local_nh = nh // TP_GROUP.size()
    local_seq_len = seq_len // TP_GROUP.size()
    input = input.reshape(bs, local_seq_len, nh, head_dim)
    a2a_buffer = torch.empty(
        (seq_len, bs, local_nh, head_dim),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    input_before_a2a = input.permute(2, 1, 0, 3).contiguous()  # [nh, local_seq_len, bs, hd]
    torch.distributed.all_to_all_single(a2a_buffer, input_before_a2a, group=TP_GROUP)

    a2a_buffer = (
        a2a_buffer.reshape(TP_GROUP.size(), local_nh, local_seq_len, bs, head_dim)
        .permute(3, 1, 0, 2, 4)
        .reshape(bs, local_nh, seq_len, head_dim)
    )
    return [a2a_buffer]


def _torch_qkv_pack_a2a(input, bs, seq_len, nh, head_dim, gqa):
    world_size = TP_GROUP.size()
    local_seq_len = seq_len // TP_GROUP.size()
    local_nh = nh // TP_GROUP.size()
    input = input.reshape(bs, local_seq_len, nh, head_dim)
    local_q_nh = local_nh // (gqa + 2) * gqa
    local_k_nh = local_nh // (gqa + 2)
    local_v_nh = local_k_nh
    q_input = input[:, :, : local_q_nh * world_size, :].contiguous()
    k_input = input[
        :, :, local_q_nh * world_size : (local_q_nh + local_k_nh) * world_size, :
    ].contiguous()
    v_input = input[:, :, (local_q_nh + local_k_nh) * world_size :, :].contiguous()

    def _a2a(a2a_tensor):
        a2a_input = a2a_tensor.permute(2, 1, 0, 3).contiguous()  # [nh, local_seq_len, bs, hd]
        a2a_nh, a2a_local_seq_len, a2a_bs, a2a_hd = a2a_input.shape
        a2a_buffer = torch.empty(
            (world_size, a2a_nh // world_size, a2a_local_seq_len, a2a_bs, a2a_hd),
            dtype=a2a_input.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        torch.distributed.all_to_all_single(a2a_buffer, a2a_input, group=TP_GROUP)
        a2a_buffer = (
            a2a_buffer.permute(3, 0, 2, 1, 4)
            .reshape(a2a_bs, a2a_local_seq_len * world_size, a2a_nh // world_size, a2a_hd)
            .contiguous()
        )
        return a2a_buffer

    q = _a2a(q_input)
    k = _a2a(k_input)
    v = _a2a(v_input)
    return [q, k, v]


def get_torch_output(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    config: TuningConfig,
    gqa: int,
    comm_op: flux.PreAttnAllToAllCommOp,
):
    bs, local_seq_len, hidden_dim = input.shape
    seq_len = local_seq_len * TP_GROUP.size()
    head_dim = config.head_dim
    gemm_n = weight.shape[0] if not config.transpose_weight else weight.shape[1]
    nh = gemm_n // head_dim

    torch.distributed.barrier()

    if not config.transpose_weight:
        gemm_output = torch.matmul(input, weight.t())
    else:
        gemm_output = torch.matmul(input, weight)
    if bias is not None:
        bias = bias.reshape(gemm_output.shape)
        gemm_output += bias

    if comm_op == flux.PreAttnAllToAllCommOp.A2ATranspose:
        outputs = _torch_all_to_all_transpose(gemm_output, bs, seq_len, nh, head_dim)
    else:
        outputs = _torch_qkv_pack_a2a(gemm_output, bs, seq_len, nh, head_dim, gqa)
    torch.distributed.barrier()
    return outputs


def _verify_and_check_bitwise(
    torch_outs: List[torch.Tensor], flux_outs: List[torch.Tensor], atol, rtol
):
    is_bitwise = True
    for ref_out, flux_out in zip(torch_outs, flux_outs):
        flux_out = flux_out.reshape(ref_out.shape)
        flux.torch_allclose(ref_out, flux_out, atol=atol, rtol=rtol)
        if not flux.bitwise_check(ref_out, flux_out):
            is_bitwise = False
    return is_bitwise


def run_flux_profiling(
    prof_ctx: flux.ProfilingContext,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    config: TuningConfig,
    gqa: int,
    comm_op: flux.PreAttnAllToAllCommOp,
):
    input_dtype = input.dtype
    bs, local_seq_len, hidden_dim = input.shape
    head_dim = config.head_dim
    seq_len = local_seq_len * TP_GROUP.size()
    gemm_m = bs * local_seq_len
    gemm_k = hidden_dim
    if config.transpose_weight:
        w = weight.t().contiguous()
        gemm_n = w.shape[1]
    else:
        w = weight
        gemm_n = w.shape[0]

    output_dtype = input.dtype
    gemm_a2a_transpose_kernel = flux.GemmAllToAllTranspose(
        TP_GROUP,
        NNODES,
        bs,
        seq_len,
        hidden_dim,
        head_dim,
        gemm_n,
        input_dtype,
        output_dtype=output_dtype,
        transpose_weight=config.transpose_weight,
        gqa=gqa,
        comm_op=comm_op,
    )

    outputs = gemm_a2a_transpose_kernel.profiling(
        input,
        w,
        bias=bias,
        input_scale=None,
        weight_scale=None,
        output_scale=None,
        fast_accum=False,
        num_comm_sm=config.num_comm_sm,
        prof_ctx=prof_ctx,
    )
    return outputs


def tune_one_config(
    prof_ctx: flux.ProfilingContext,
    config: TuningConfig,
    gqa: int,
    comm_op: flux.PreAttnAllToAllCommOp,
):
    assert config.seq % TP_GROUP.size() == 0
    assert config.out_features % (TP_GROUP.size() * config.head_dim) == 0

    local_seq = config.seq // TP_GROUP.size()
    input_shape = [config.bs, local_seq, config.hidden_dim]
    weight_shape = [config.out_features, config.hidden_dim]

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

    torch_outputs = get_torch_output(input, weight, bias, config, gqa, comm_op)
    torch.cuda.current_stream().synchronize()
    torch.distributed.barrier()
    flux_outputs = run_flux_profiling(prof_ctx, input, weight, bias, config, gqa, comm_op)
    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    if config.dtype == torch.bfloat16:
        atol, rtol = 0.02, 0.02
    else:
        atol, rtol = 0.01, 0.01
    _verify_and_check_bitwise(torch_outputs, flux_outputs, atol=atol, rtol=rtol)
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
        "--comm_op",
        default="A2ATranspose",
        choices=["A2ATranspose", "QKVPackA2A"],
        help="pre attn all to all communication operation",
    )
    parser.add_argument("--gqa", default=0, type=int, help="group size of group query attn")
    return parser.parse_args()


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "s8": torch.int8,
}


COMM_OP_MAP = {
    "A2ATranspose": flux.PreAttnAllToAllCommOp.A2ATranspose,
    "QKVPackA2A": flux.PreAttnAllToAllCommOp.QKVPackA2A,
}


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir and not os.path.isdir(args.output_dir):
        raise Exception(f"{args.output_dir} not exist")

    torch.cuda.set_device(LOCAL_RANK)
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()

    if args.comm_op == "QKVPackA2A" and args.gqa < 1:
        raise ValueError("gqa must be greater than 0 for QKVPackA2A")
    if args.comm_op == "A2ATranspose" and args.gqa != 0:
        raise ValueError("gqa must be equal to 0 for A2ATranspose")
    gqa = args.gqa
    comm_op = COMM_OP_MAP[args.comm_op]

    arch: int = flux.get_arch()
    name: str = f"config_gemm_a2a_transpose_kernel_sm{arch}_tp{WORLD_SIZE}_nnodes{NNODES}"
    prof_ctx = flux.ProfilingContext(name)
    dtype = DTYPE_MAP[args.dtype]
    config_space = gen_tuning_space(dtype, args.check)
    for i, config in enumerate(config_space):
        if TP_GROUP.rank() == 0:
            print(f"==== #{i + 1}/{len(config_space)} Tuning for {config}")
        tune_one_config(prof_ctx=prof_ctx, config=config, gqa=gqa, comm_op=comm_op)
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
