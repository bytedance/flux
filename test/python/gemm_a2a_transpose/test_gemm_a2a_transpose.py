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
        gemm_output: torch.Tensor,
        total_ms: float,
        time1: str,
        gemm_time_ms: float,
        time2: str,
        comm_time_ms: float,
        time3: str = "gemm_only",
        gemm_only_time_ms: float = 0,
    ) -> None:
        self.name = name
        self.outputs = outputs
        self.gemm_output = gemm_output
        self.total_ms = total_ms
        self.time1 = time1
        self.time2 = time2
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms
        self.time3 = time3
        self.gemm_only_time_ms = gemm_only_time_ms

    def __repr__(self) -> str:
        if self.gemm_only_time_ms == 0.0:
            return (
                f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms"
            )
        else:
            return (
                f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms, {self.time3} {self.gemm_only_time_ms:.3f} ms"
            )


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
    bs, max_seq_len, hidden_dim, head_dim = args.bs, args.seq_len, args.hidden_dim, args.head_dim
    max_local_seq_len = max_seq_len // sp_group.size()
    out_features = args.out_features

    gemm_k = hidden_dim
    gemm_n = out_features

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
        hidden_dim,
        head_dim,
        gemm_n,
        input_dtype,
        output_dtype=output_dtype,
        transpose_weight=args.transpose_weight,
        gqa=gqa,
        comm_op=comm_op,
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

        input_shape = [bs, local_seq_len, gemm_k]
        weight_shape = [gemm_n, gemm_k]

        input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) / 10 * (sp_group.rank() + 1)
        weight = (
            (-2 * torch.rand(weight_shape, dtype=dtype).cuda() + 1) / 10 * (sp_group.rank() + 1)
        )

        if is_debug:
            input.fill_(iter % 2 + 1)
            weight.fill_(iter % 2 + 1)

        input_scale = None
        weight_scale = None

        bias = None
        gemm_m = bs * local_seq_len
        bias_shape = [gemm_m, gemm_n]
        if args.has_bias:
            bias = torch.rand(bias_shape, dtype=dtype).cuda() / 10 * (sp_group.rank() + 1)

        return (input, weight, seq_lens_cpu, input_scale, weight_scale, bias)

    def _torch_impl(input, weight, seq_lens_cpu, input_scale, weight_scale, bias):
        gemm_output = torch.matmul(input, weight.t())

        if bias is not None:
            gemm_output += bias
        seq_len = input.shape[1] * sp_group.size()
        if comm_op == flux.PreAttnAllToAllCommOp.A2ATranspose:
            outputs = torch_pre_attn_all_to_all_transpose(
                sp_group, gemm_output, bs, seq_len, nh, head_dim, seq_lens_cpu
            )
        else:
            outputs = torch_pre_attn_qkv_pack_a2a(
                sp_group, gemm_output, bs, seq_len, nh, head_dim, gqa, seq_lens_cpu
            )
        return outputs

    def _flux_impl(input, weight, seq_lens_cpu, input_scale, weight_scale, bias):
        outputs_buf = None
        if args.use_external_outputs:
            num_total_heads = gemm_n // head_dim
            bs = input.shape[0]
            sp_size = sp_group.size()
            seq_len = (
                seq_lens_cpu.sum().item() if seq_lens_cpu != None else input.shape[1] * sp_size
            )
            if args.comm_op == "QKVPackA2A":
                assert num_total_heads % (gqa + 2) == 0
                num_q_heads = num_total_heads // (gqa + 2) * gqa
                num_kv_heads = num_total_heads // (gqa + 2)
                q_buf = torch.empty(
                    (bs, seq_len, num_q_heads // sp_size, head_dim),
                    dtype=output_dtype,
                    device=input.device,
                )
                k_buf = torch.empty(
                    (bs, seq_len, num_kv_heads // sp_size, head_dim),
                    dtype=output_dtype,
                    device=input.device,
                )
                v_buf = torch.empty(
                    (bs, seq_len, num_kv_heads // sp_size, head_dim),
                    dtype=output_dtype,
                    device=input.device,
                )
                outputs_buf = [q_buf, k_buf, v_buf]
            else:
                q_buf = torch.empty(
                    (bs, num_total_heads // sp_size, seq_len, head_dim),
                    dtype=output_dtype,
                    device=input.device,
                )
                outputs_buf = [q_buf]

        outputs = flux_gemm_a2a_transpose_kernel.forward(
            input,
            weight,
            seq_lens_cpu=seq_lens_cpu,
            bias=bias,
            outputs=outputs_buf,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=args.fastacc,
            num_comm_sm=args.num_comm_sm,
            sm_margin=args.sm_margin,
        )
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

        atol = THRESHOLD_MAP[dtype]
        rtol = THRESHOLD_MAP[dtype]

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
    weight: torch.Tensor,
    seq_lens_cpu: Optional[torch.Tensor],
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    head_dim: int,
    transpose_weight: bool,
    warmup: int,
    iters: int,
    gqa: int = 0,
    comm_op: flux.PreAttnAllToAllCommOp = flux.PreAttnAllToAllCommOp.A2ATranspose,
):
    gemm_n = weight.shape[0] if not transpose_weight else weight.shape[1]
    bs, local_seq_len, hidden_dim = input.shape
    seq_len = (
        local_seq_len * sp_group.size() if seq_lens_cpu == None else sum(seq_lens_cpu.tolist())
    )
    nh = gemm_n // head_dim
    # All to all input tensors from all gpus

    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    all2all_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        if not transpose_weight:
            gemm_output = torch.matmul(input, weight.t())
        else:
            gemm_output = torch.matmul(input, weight)
        if bias is not None:
            bias = bias.reshape(gemm_output.shape)
            gemm_output += bias

        gemm_end_events[i].record()

        if comm_op == flux.PreAttnAllToAllCommOp.A2ATranspose:
            outputs = torch_pre_attn_all_to_all_transpose(
                sp_group, gemm_output, bs, seq_len, nh, head_dim, seq_lens_cpu
            )
        else:
            outputs = torch_pre_attn_qkv_pack_a2a(
                sp_group, gemm_output, bs, seq_len, nh, head_dim, gqa, seq_lens_cpu
            )
        all2all_end_events[i].record()

    comm_times = []  # all to all
    gemm_times = []  # gemm
    for i in range(total_iters):
        gemm_end_events[i].synchronize()
        all2all_end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_times.append(start_events[i].elapsed_time(gemm_end_events[i]) / 1000)
            comm_times.append(gemm_end_events[i].elapsed_time(all2all_end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        outputs=outputs,
        gemm_output=gemm_output,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_flux(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    weight: torch.Tensor,
    seq_lens_cpu: Optional[torch.Tensor],
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    head_dim: int,
    transpose_weight: bool = True,
    warmup: int = 5,
    iters: int = 10,
    num_comm_sm: int = -1,
    sm_margin: int = 0,
    fast_acc: bool = False,
    gqa: int = 0,
    comm_op: flux.PreAttnAllToAllCommOp = flux.PreAttnAllToAllCommOp.A2ATranspose,
):
    input_dtype = input.dtype
    bs, local_seq_len, hidden_dim = input.shape

    max_local_seq_len = (
        (max(seq_lens_cpu.tolist()) + 127) // 128 * 128 if seq_lens_cpu != None else local_seq_len
    )
    max_seq_len = max_local_seq_len * sp_group.size()
    gemm_m = bs * local_seq_len
    gemm_k = hidden_dim
    if transpose_weight:
        w = weight.t().contiguous()
        gemm_n = w.shape[1]
    else:
        w = weight
        gemm_n = w.shape[0]

    nh = gemm_n // head_dim

    torch.distributed.barrier()

    output_dtype = input_dtype
    gemm_only_op = flux.GemmOnly(
        input_dtype=input_dtype,
        weight_dtype=input_dtype,
        output_dtype=output_dtype,
        transpose_weight=transpose_weight,
        use_fp8_gemm=False,
    )

    gemm_only_output = torch.empty(
        [gemm_m, gemm_n],
        dtype=output_dtype,
        device=input.device,
        requires_grad=False,
    )

    gemm_a2a_transpose_kernel = flux.GemmAllToAllTranspose(
        TP_GROUP,  # global process group
        NNODES,
        sp_group.size(),
        bs,
        max_seq_len,
        hidden_dim,
        head_dim,
        gemm_n,
        input_dtype,
        output_dtype=output_dtype,
        transpose_weight=transpose_weight,
        gqa=gqa,
        comm_op=comm_op,
    )

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        gemm_only_output = gemm_only_op.forward(
            input.reshape(gemm_m, gemm_k),
            w,
            bias=bias,
            output_buf=gemm_only_output,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
        )
        end_events[i].record()
    torch.cuda.current_stream().synchronize()

    gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)
    gemm_time = sum(gemm_times)

    time.sleep(1)

    torch.distributed.barrier()

    for i in range(total_iters):
        start_events[i].record()
        outs = gemm_a2a_transpose_kernel.forward(
            input,
            w,
            seq_lens_cpu=seq_lens_cpu,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            num_comm_sm=num_comm_sm,
            sm_margin=sm_margin,
        )
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    a2a_gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            a2a_gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    a2a_gemm_time = sum(a2a_gemm_times)

    a2a_gemm_time_ms = a2a_gemm_time / iters * 1000
    gemm_time_ms = gemm_time / iters * 1000
    comm_time_ms = (a2a_gemm_time - gemm_time) / iters * 1000

    return PerfResult(
        name=f"flux  #{TP_GROUP.rank()}",
        outputs=outs,
        gemm_output=gemm_only_output,
        total_ms=a2a_gemm_time_ms,
        time1="gemm",
        gemm_time_ms=gemm_time_ms,
        time2="comm",
        comm_time_ms=comm_time_ms,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bs", type=int)
    parser.add_argument("seq_len", type=int)
    parser.add_argument("hidden_dim", type=int)
    parser.add_argument("head_dim", type=int)
    parser.add_argument("out_features", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument("--num_comm_sm", type=int, required=True, help="num sm for a2a")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument(
        "--fastacc",
        default=False,
        action="store_true",
        help="whether to use fast accumulation (FP8 Gemm only)",
    )
    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=False,
    )
    parser.add_argument(
        "--use_external_outputs",
        default=False,
        action="store_true",
        help="alloc outputs buffer externally",
    )
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run once to verify correctness",
    )
    parser.add_argument("--dp", default=False, action="store_true", help="dp per rank")
    parser.add_argument(
        "--comm_op",
        default="A2ATranspose",
        choices=["A2ATranspose", "QKVPackA2A"],
        help="pre attn all to all communication operation",
    )
    parser.add_argument("--gqa", default=0, type=int, help="group size of group query attn")
    parser.add_argument("--sp_size", default=0, type=int, help="sp size")
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
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

    if args.transpose_weight:
        raise NotImplemented("GemmAllToAllTranspose only support RCR layout.")

    if args.comm_op == "QKVPackA2A" and args.gqa < 1:
        raise ValueError("gqa must be greater than 0 for QKVPackA2A")
    if args.comm_op == "A2ATranspose" and args.gqa != 0:
        raise ValueError("gqa must be equal to 0 for A2ATranspose")
    if args.dp and args.comm_op != "QKVPackA2A":
        raise NotImplementedError("only QKVPackA2A support dp")

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

    assert args.out_features % args.head_dim == 0
    assert args.hidden_dim % args.head_dim == 0

    nh = args.out_features // args.head_dim
    assert nh % sp_group.size() == 0
    assert args.seq_len % sp_group.size() == 0
    np.random.seed(3 + RANK // args.sp_size)

    max_local_seq_len = args.seq_len // sp_group.size()
    hidden_dim = args.hidden_dim
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
    input_shape = [args.bs, local_seq_len, hidden_dim]
    weight_shape = [args.out_features, hidden_dim]

    input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) / 10 * (sp_group.rank() + 1)
    weight = (-2 * torch.rand(weight_shape, dtype=dtype).cuda() + 1) / 10 * (sp_group.rank() + 1)

    input_scale = None
    weight_scale = None

    bias = None
    gemm_m = args.bs * local_seq_len
    bias_shape = [gemm_m, args.out_features]
    if args.has_bias:
        bias = torch.rand(bias_shape, dtype=dtype).cuda() / 10 * (sp_group.rank() + 1)

    if args.debug:
        input.zero_()
        input[:, 0].fill_(sp_group.rank() + 1)
        weight.fill_(1)
        if input_scale is not None:
            input_scale.fill_(1)
            weight_scale.fill_(1)
        if bias is not None:
            bias.zero_()
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
            weight,
            seq_lens_cpu,
            bias,
            input_scale,
            weight_scale,
            args.head_dim,
            args.transpose_weight,
            args.warmup,
            args.iters,
            args.gqa,
            COMM_OP_MAP[args.comm_op],
        )
        perf_res_flux = perf_flux(
            sp_group,
            input,
            weight,
            seq_lens_cpu,
            bias,
            input_scale,
            weight_scale,
            args.head_dim,
            args.transpose_weight,
            args.warmup,
            args.iters,
            args.num_comm_sm,
            args.sm_margin,
            args.fastacc,
            args.gqa,
            COMM_OP_MAP[args.comm_op],
        )

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    if TP_GROUP.rank() == 0:
        gemm_m = args.bs * local_seq_len
        gemm_n = args.out_features
        gemm_k = args.hidden_dim
        flux.testing.print_gemm_sol_time(gemm_m, gemm_n, gemm_k, dtype)

    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            print(perf_res_torch)
            print(perf_res_flux)
        torch.distributed.barrier()

    torch_outputs = perf_res_torch.outputs
    flux_outputs = perf_res_flux.outputs

    torch.cuda.synchronize()
    torch.distributed.barrier()

    atol = THRESHOLD_MAP[dtype]
    rtol = THRESHOLD_MAP[dtype]
    is_bitwise = _verify_and_check_bitwise(torch_outputs, flux_outputs, atol=atol, rtol=rtol)
    if is_bitwise:
        print("✅  torch vs flux bitwise match")
    else:
        print("❌  torch vs flux not bitwise match")
    TP_GROUP.barrier()
    torch.cuda.synchronize()
