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
from typing import List, Optional

import random
import numpy as np
import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import torch_pre_attn_all_to_all_transpose, torch_pre_attn_qkv_pack_a2a

print = partial(print, flush=True)

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

torch.cuda.set_device(LOCAL_RANK)
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(False, warn_only=True)
torch.set_printoptions(precision=2)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
np.random.seed(3)


torch.distributed.init_process_group(
    backend="nccl", world_size=WORLD_SIZE, rank=RANK, timeout=datetime.timedelta(seconds=1800)
)
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")


class PerfResult:
    def __init__(
        self,
        name: str,
        outputs: List[torch.Tensor],
        total_ms: float,
    ) -> None:
        self.name = name
        self.outputs = outputs
        self.total_ms = total_ms

    def __repr__(self) -> str:
        return f"{self.name}: total {self.total_ms:.3f} ms"


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


def check_correctness(sp_group, args):
    random.seed(42 + RANK // sp_group.size())

    num_iteration = args.iters
    bs, max_seq_len, nh, head_dim = args.bs, args.seq_len, args.nheads, args.head_dim
    max_local_seq_len = max_seq_len // sp_group.size()
    out_features = nh * head_dim

    dtype = DTYPE_MAP[args.dtype]
    input_dtype = dtype
    output_dtype = dtype

    comm_op = COMM_OP_MAP[args.comm_op]
    gqa = args.gqa
    flux_gemm_a2a_transpose_kernel = flux.GemmAllToAllTranspose(
        TP_GROUP,
        NNODES,
        sp_group.size(),
        bs,
        max_seq_len,
        out_features,
        head_dim,
        out_features,
        input_dtype,
        output_dtype=output_dtype,
        gqa=gqa,
        comm_op=comm_op,
        max_num_comm_buf=1 if args.apply_pack else 3,
    )

    def _gen_inputs(max_local_seq_len, iter=0, is_debug=False):
        if not args.dp:
            seq_lens_cpu = None
            local_seq_len = max_local_seq_len
        else:
            seq_lens_list = list(
                np.random.randint(
                    max(1, max_local_seq_len - 32), max_local_seq_len, size=(sp_group.size(),)
                )
            )
            local_seq_len = seq_lens_list[sp_group.rank()]
            seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)
            if sp_group.rank() == 0:
                print(f"max_local_seq_len = {max_local_seq_len}, seq_lens_cpu = {seq_lens_cpu}")

        input_shape = [bs, local_seq_len, args.nheads, args.head_dim]
        input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) / 10 * (sp_group.rank() + 1)
        return (input, seq_lens_cpu)

    def _torch_impl(input, seq_lens_cpu):
        seq_len = input.size(1) * sp_group.size() if seq_lens_cpu == None else seq_lens_cpu.sum()
        if comm_op == flux.PreAttnAllToAllCommOp.A2ATranspose:
            outputs = torch_pre_attn_all_to_all_transpose(
                sp_group, input, bs, seq_len, nh, head_dim, seq_lens_cpu
            )
        else:
            outputs = torch_pre_attn_qkv_pack_a2a(
                sp_group, input, bs, seq_len, nh, head_dim, gqa, seq_lens_cpu
            )
        return outputs

    def _flux_impl(input, seq_lens_cpu):
        if not args.local_copy:
            if args.apply_pack:
                outputs = flux_gemm_a2a_transpose_kernel.pre_attn_qkv_pack_a2a(
                    input,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                )
            else:
                q_nh = nh // (gqa + 2) * gqa
                k_nh = nh // (gqa + 2)
                q = input[:, :, :q_nh].contiguous()
                k = input[:, :, q_nh : q_nh + k_nh].contiguous()
                v = input[:, :, q_nh + k_nh :].contiguous()
                out_q = flux_gemm_a2a_transpose_kernel.pre_attn_a2a(
                    q,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                )
                out_k = flux_gemm_a2a_transpose_kernel.pre_attn_a2a(
                    k,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                )
                out_v = flux_gemm_a2a_transpose_kernel.pre_attn_a2a(
                    v,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                )
                outputs = [out_q, out_k, out_v]
        else:
            if args.apply_pack:
                comm_buf = flux_gemm_a2a_transpose_kernel.get_input_comm_buf(input, 0)
                comm_buf.copy_(input)
                flux_gemm_a2a_transpose_kernel.sp_group_barrier_all()
                outputs = flux_gemm_a2a_transpose_kernel.pre_attn_qkv_pack_a2a_no_cpy(
                    input,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                    comm_buf_idx=0,
                )
                flux_gemm_a2a_transpose_kernel.sp_group_barrier_all()
            else:
                q_nh = nh // (gqa + 2) * gqa
                k_nh = nh // (gqa + 2)
                q = input[:, :, :q_nh].contiguous()
                k = input[:, :, q_nh : q_nh + k_nh].contiguous()
                v = input[:, :, q_nh + k_nh :].contiguous()
                q_comm_buf = flux_gemm_a2a_transpose_kernel.get_input_comm_buf(q, 0)
                k_comm_buf = flux_gemm_a2a_transpose_kernel.get_input_comm_buf(k, 1)
                v_comm_buf = flux_gemm_a2a_transpose_kernel.get_input_comm_buf(v, 2)
                q_comm_buf.copy_(q)
                k_comm_buf.copy_(k)
                v_comm_buf.copy_(v)
                flux_gemm_a2a_transpose_kernel.sp_group_barrier_all()
                out_q = flux_gemm_a2a_transpose_kernel.pre_attn_a2a_no_cpy(
                    q,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                    comm_buf_idx=0,
                )
                out_k = flux_gemm_a2a_transpose_kernel.pre_attn_a2a_no_cpy(
                    k,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                    comm_buf_idx=1,
                )
                out_v = flux_gemm_a2a_transpose_kernel.pre_attn_a2a_no_cpy(
                    v,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=args.num_comm_sm,
                    comm_buf_idx=2,
                )
                flux_gemm_a2a_transpose_kernel.sp_group_barrier_all()
                outputs = [out_q, out_k, out_v]
        return outputs

    all_inputs = [
        _gen_inputs(random.randint(1, max_local_seq_len), idx) for idx in range(num_iteration)
    ]
    torch.cuda.synchronize()
    torch.distributed.barrier()

    all_torch_outputs = [_torch_impl(*inputs) for inputs in all_inputs]

    torch.cuda.synchronize()
    torch.distributed.barrier()

    all_flux_outputs = [_flux_impl(*inputs) for inputs in all_inputs]

    torch.cuda.synchronize()
    torch.distributed.barrier()

    is_bitwise = True
    for idx, (flux_outs, torch_outs) in enumerate(zip(all_torch_outputs, all_flux_outputs)):

        atol, rtol = 0, 0
        if not _verify_and_check_bitwise(torch_outs, flux_outs, atol, rtol):
            is_bitwise = False

    if is_bitwise:
        print(f"rank[{TP_GROUP.rank()}]: ✅  torch vs flux bitwise match")
    else:
        print(f"rank[{TP_GROUP.rank()}]: ❌  torch vs flux not bitwise match")
    TP_GROUP.barrier()
    torch.cuda.synchronize()


@torch.no_grad()
def perf_torch(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    seq_lens_cpu: Optional[torch.Tensor],
    warmup: int,
    iters: int,
    gqa: int = 0,
    comm_op: flux.PreAttnAllToAllCommOp = flux.PreAttnAllToAllCommOp.A2ATranspose,
):
    bs, local_seq_len, nh, head_dim = input.shape
    seq_len = (
        local_seq_len * sp_group.size() if seq_lens_cpu == None else sum(seq_lens_cpu.tolist())
    )
    # All to all input tensors from all gpus

    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        if comm_op == flux.PreAttnAllToAllCommOp.A2ATranspose:
            outputs = torch_pre_attn_all_to_all_transpose(
                sp_group, input, bs, seq_len, nh, head_dim, seq_lens_cpu
            )
        else:
            outputs = torch_pre_attn_qkv_pack_a2a(
                sp_group, input, bs, seq_len, nh, head_dim, gqa, seq_lens_cpu
            )
        end_events[i].record()

    comm_times = []  # all to all
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        outputs=outputs,
        total_ms=comm_time,
    )


@torch.no_grad()
def perf_flux(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    seq_lens_cpu: Optional[torch.Tensor],
    warmup: int = 5,
    iters: int = 10,
    num_comm_sm: int = -1,
    gqa: int = 0,
    comm_op: flux.PreAttnAllToAllCommOp = flux.PreAttnAllToAllCommOp.A2ATranspose,
    apply_pack: bool = True,
    local_copy: bool = False,
):
    input_dtype = input.dtype
    bs, local_seq_len, nh, head_dim = input.shape
    q_nh = nh // (gqa + 2) * gqa
    k_nh = nh // (gqa + 2)
    v_nh = nh // (gqa + 2)
    q = input[:, :, :q_nh].contiguous()
    k = input[:, :, q_nh : q_nh + k_nh].contiguous()
    v = input[:, :, q_nh + k_nh :].contiguous()
    max_local_seq_len = (
        (max(seq_lens_cpu.tolist()) + 127) // 128 * 128 if seq_lens_cpu != None else local_seq_len
    )
    max_seq_len = max_local_seq_len * sp_group.size()
    output_dtype = input_dtype

    gemm_a2a_transpose_kernel = flux.GemmAllToAllTranspose(
        TP_GROUP,  # global process group
        NNODES,
        sp_group.size(),
        bs,
        max_seq_len,
        nh * head_dim,
        head_dim,
        nh * head_dim,
        input_dtype,
        output_dtype=output_dtype,
        gqa=gqa,
        comm_op=comm_op,
        max_num_comm_buf=1 if apply_pack else 3,
    )

    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    time.sleep(1)

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        if not local_copy:
            if apply_pack:
                outs = gemm_a2a_transpose_kernel.pre_attn_qkv_pack_a2a(
                    input,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                )
                print(f"outs = {outs}")
            else:
                out_q = gemm_a2a_transpose_kernel.pre_attn_a2a(
                    q,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                )
                out_k = gemm_a2a_transpose_kernel.pre_attn_a2a(
                    k,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                )
                out_v = gemm_a2a_transpose_kernel.pre_attn_a2a(
                    v,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                )
                outs = [out_q, out_k, out_v]
        else:
            if apply_pack:
                comm_buf = gemm_a2a_transpose_kernel.get_input_comm_buf(input, 0)
                comm_buf.copy_(input)
                gemm_a2a_transpose_kernel.sp_group_barrier_all()
                outs = gemm_a2a_transpose_kernel.pre_attn_qkv_pack_a2a_no_cpy(
                    input, seq_lens_cpu=seq_lens_cpu, num_comm_sm=num_comm_sm, comm_buf_idx=0
                )
                gemm_a2a_transpose_kernel.sp_group_barrier_all()
            else:
                q_comm_buf = gemm_a2a_transpose_kernel.get_input_comm_buf(q, 0)
                k_comm_buf = gemm_a2a_transpose_kernel.get_input_comm_buf(k, 1)
                v_comm_buf = gemm_a2a_transpose_kernel.get_input_comm_buf(v, 2)
                q_comm_buf.copy_(q)
                k_comm_buf.copy_(k)
                v_comm_buf.copy_(v)
                gemm_a2a_transpose_kernel.sp_group_barrier_all()
                out_q = gemm_a2a_transpose_kernel.pre_attn_a2a_no_cpy(
                    q,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                    comm_buf_idx=0,
                )
                out_k = gemm_a2a_transpose_kernel.pre_attn_a2a_no_cpy(
                    k,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                    comm_buf_idx=1,
                )
                out_v = gemm_a2a_transpose_kernel.pre_attn_a2a_no_cpy(
                    v,
                    seq_lens_cpu=seq_lens_cpu,
                    num_comm_sm=num_comm_sm,
                    comm_buf_idx=2,
                )
                gemm_a2a_transpose_kernel.sp_group_barrier_all()
                outs = [out_q, out_k, out_v]
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    a2a_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            a2a_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    a2a_time = sum(a2a_times)

    a2a_time_ms = a2a_time / iters * 1000

    return PerfResult(name=f"flux  #{TP_GROUP.rank()}", outputs=outs, total_ms=a2a_time_ms)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bs", type=int)
    parser.add_argument("seq_len", type=int)
    parser.add_argument("nheads", type=int)
    parser.add_argument("head_dim", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--num_comm_sm", type=int, required=True, help="num sm for a2a")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="verify correctness",
    )
    parser.add_argument("--dp", default=False, action="store_true", help="dp per rank")
    parser.add_argument(
        "--comm_op",
        default="QKVPackA2A",
        choices=["QKVPackA2A"],
        help="pre attn all to all communication operation",
    )
    parser.add_argument(
        "--local-copy",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If local-copy is true, the user needs to ensure that the input is copied to comm buffer",
    )
    parser.add_argument("--gqa", default=0, type=int, help="group size of group query attn")
    parser.add_argument("--sp_size", default=0, required=True, type=int, help="sp size")
    parser.add_argument(
        "--apply_pack",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to pack q/k/v",
    )
    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

COMM_OP_MAP = {
    "A2ATranspose": flux.PreAttnAllToAllCommOp.A2ATranspose,
    "QKVPackA2A": flux.PreAttnAllToAllCommOp.QKVPackA2A,
}

if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    args = parse_args()

    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()

    dtype = DTYPE_MAP[args.dtype]

    if dtype not in [torch.bfloat16]:
        raise NotImplementedError("GemmAllToAllTranspose only support BF16.")

    if args.comm_op != "QKVPackA2A":
        raise NotImplementedError("only QKVPackA2A supported")

    # init sp process group
    assert args.sp_size > 0 and LOCAL_WORLD_SIZE % args.sp_size == 0
    num_sp_group = WORLD_SIZE // args.sp_size
    all_sp_subgroups = []
    sp_group = None
    for i in range(num_sp_group):
        cur_group_ranks = [i * args.sp_size + j for j in range(args.sp_size)]
        all_sp_subgroups.append(torch.distributed.new_group(cur_group_ranks))
        if i == RANK // args.sp_size:
            sp_group = all_sp_subgroups[-1]
    assert sp_group != None
    assert args.seq_len % sp_group.size() == 0
    np.random.seed(3 + RANK // args.sp_size)

    max_local_seq_len = args.seq_len // sp_group.size()
    if not args.dp:
        seq_lens_cpu = None
        local_seq_len = max_local_seq_len
    else:
        seq_lens_list = list(
            np.random.randint(
                max(1, max_local_seq_len - 32),
                max_local_seq_len,
                size=(sp_group.size(),),
            )
        )
        local_seq_len = seq_lens_list[sp_group.rank()]
        seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)
        if sp_group.rank() == 0:
            print(
                f"sp_group_id = {TP_GROUP.rank() // args.sp_size}, seq_lens_list = {seq_lens_list}"
            )
    input_shape = [args.bs, local_seq_len, args.nheads, args.head_dim]
    input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) / 10 * (sp_group.rank() + 1)

    torch.distributed.barrier()

    ctx = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        )
        if args.profile
        else nullcontext()
    )

    if args.verify:
        check_correctness(sp_group, args)
        exit(0)

    with ctx:
        perf_res_torch = perf_torch(
            sp_group,
            input,
            seq_lens_cpu,
            args.warmup,
            args.iters,
            args.gqa,
            COMM_OP_MAP[args.comm_op],
        )
        perf_res_flux = perf_flux(
            sp_group,
            input,
            seq_lens_cpu,
            args.warmup,
            args.iters,
            args.num_comm_sm,
            args.gqa,
            COMM_OP_MAP[args.comm_op],
            args.apply_pack,
            args.local_copy,
        )

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            print(perf_res_torch)
            print(perf_res_flux)
        torch.distributed.barrier()

    torch_outputs = perf_res_torch.outputs
    flux_outputs = perf_res_flux.outputs

    torch.cuda.synchronize()
    torch.distributed.barrier()

    atol, rtol = 0, 0
    is_bitwise = _verify_and_check_bitwise(torch_outputs, flux_outputs, atol=atol, rtol=rtol)
    if is_bitwise:
        print("✅  torch vs flux bitwise match")
    else:
        print("❌  torch vs flux not bitwise match")
    TP_GROUP.barrier()
    torch.cuda.synchronize()
