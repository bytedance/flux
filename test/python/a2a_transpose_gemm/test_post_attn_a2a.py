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
from flux.testing import torch_post_attn_all_to_all_transpose

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
        a2a_output: torch.Tensor,
        total_ms: float,
    ) -> None:
        self.name = name
        self.a2a_output = a2a_output
        self.total_ms = total_ms

    def __repr__(self) -> str:
        return f"{self.name}: total {self.total_ms:.3f} ms"


RING_MODE_MAP = {
    "auto": None,
    "all2all": flux.A2ARingMode.All2All,
    "ring1d": flux.A2ARingMode.Ring1D,
    "ring2d": flux.A2ARingMode.Ring2D,
}


def check_correctness(sp_group, args):
    random.seed(42 + RANK // sp_group.size())

    num_iteration = args.iters
    bs, nh, max_seq_len, hd = args.bs, args.nh, args.seq_len, args.hd
    max_local_seq_len = args.seq_len // sp_group.size()
    dtype = DTYPE_MAP[args.dtype]

    flux_all_to_all_gemm_kernel = flux.AllToAllTransposeGemm(
        TP_GROUP,
        NNODES,
        sp_group.size(),
        bs,
        nh,
        max_seq_len,
        hd,
        dtype,
        output_dtype=dtype,
        max_num_comm_buf=1,
        a2a_only=args.a2a_only,
    )

    def _gen_inputs(max_local_seq_len):
        if not args.dp:
            seq_lens_cpu = None
            total_seq_len = max_local_seq_len * sp_group.size()
        else:
            seq_lens_list = list(
                np.random.randint(
                    max_local_seq_len // 2, max_local_seq_len, size=(sp_group.size(),)
                )
            )
            total_seq_len = sum(seq_lens_list)
            seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)
            if sp_group.rank() == 0:
                print(f"sp_group id = {RANK // sp_group.size()}, seq_lens_list = {seq_lens_list}")
        if not args.a2a_only:
            input_shape = [args.bs, local_nh, total_seq_len, args.hd]
        else:
            input_shape = [args.bs, total_seq_len, local_nh, args.hd]

        input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) * (sp_group.rank() + 1)

        return (input, seq_lens_cpu)

    def _torch_impl(input, seq_lens_cpu):
        a2a_output = torch_post_attn_all_to_all_transpose(
            sp_group, input, args.a2a_only, args.dp, seq_lens_cpu=seq_lens_cpu
        )
        return a2a_output

    def _flux_impl(input, seq_lens_cpu):
        option = flux.AllToAllOption()
        option.fuse_sync = args.fuse_sync
        if not args.local_copy:
            output = flux_all_to_all_gemm_kernel.post_attn_a2a(
                input,
                seq_lens_cpu=seq_lens_cpu,
                all_to_all_option=option,
                num_comm_sm=args.num_comm_sm,
            )
        else:
            output = flux_all_to_all_gemm_kernel.post_attn_a2a_no_cpy(
                input,
                seq_lens_cpu=seq_lens_cpu,
                all_to_all_option=option,
                num_comm_sm=args.num_comm_sm,
                comm_buf_idx=0,
            )
            flux_all_to_all_gemm_kernel.sp_group_barrier_all()
            new_output = torch.empty(output.shape, dtype=output.dtype, device=output.device)
            new_output.copy_(output)
            return new_output
        return output

    all_inputs = [_gen_inputs(random.randint(1, max_local_seq_len)) for _ in range(num_iteration)]
    torch_outputs = [_torch_impl(*inputs) for inputs in all_inputs]

    torch.distributed.barrier()
    torch.cuda.synchronize()

    flux_outputs = [_flux_impl(*inputs) for inputs in all_inputs]

    torch.distributed.barrier()

    torch.cuda.synchronize()

    for flux_output, torch_output in zip(flux_outputs, torch_outputs):
        flux_output = flux_output.reshape(torch_output.shape)
        if not flux.bitwise_check(torch_output, flux_output):
            print("Warning: torch vs flux not bitwise match")

        atol = THRESHOLD_MAP[dtype]
        rtol = THRESHOLD_MAP[dtype]
        try:
            torch.testing.assert_close(flux_output, torch_output, atol=atol, rtol=rtol)
        except Exception as e:
            torch.save(flux_output, f"flux_{TP_GROUP.rank()}.pt")
            torch.save(torch_output, f"torch_{TP_GROUP.rank()}.pt")
            print("❌ flux check failed")
            raise e

    print("✅ flux check passed")

    TP_GROUP.barrier()
    torch.cuda.synchronize()


@torch.no_grad()
def perf_torch(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    warmup: int,
    iters: int,
    a2a_only: bool = False,
    is_dp: bool = False,
    seq_lens_cpu: Optional[torch.Tensor] = None,
):
    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    all2all_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        a2a_output = torch_post_attn_all_to_all_transpose(
            sp_group, input, a2a_only, is_dp, seq_lens_cpu=seq_lens_cpu
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
        name=f"torch #{TP_GROUP.rank()}",
        a2a_output=a2a_output,
        total_ms=comm_time,
    )


@torch.no_grad()
def perf_flux(
    sp_group: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    warmup: int = 5,
    iters: int = 10,
    num_comm_sm: int = -1,
    fuse_sync: bool = False,
    a2a_only: bool = False,
    is_dp: bool = False,
    local_copy: bool = False,
    seq_lens_cpu: Optional[torch.Tensor] = None,
):
    if not is_dp:
        assert seq_lens_cpu == None

    input_dtype = input.dtype
    if not a2a_only:
        bs, local_nh, seq_len, hd = input.shape
    else:
        bs, seq_len, local_nh, hd = input.shape
    nh = local_nh * sp_group.size()
    local_seq_len = seq_len // sp_group.size()
    hidden_dim = local_nh * hd * sp_group.size()
    if is_dp:
        local_seq_len = seq_lens_cpu[sp_group.rank()].item()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    output_dtype = input_dtype

    option = flux.AllToAllOption()
    option.fuse_sync = fuse_sync
    max_seq_len = (
        (max(seq_lens_cpu.tolist()) + 127) // 128 * 128 * sp_group.size() if is_dp else seq_len
    )
    all_to_all_gemm_kernel = flux.AllToAllTransposeGemm(
        TP_GROUP,
        NNODES,
        sp_group.size(),
        bs,
        nh,
        max_seq_len,  # max_local_seq_len * world_size
        hd,
        input_dtype,
        output_dtype=output_dtype,
        a2a_only=a2a_only,
    )
    torch.distributed.barrier()

    for i in range(total_iters):
        start_events[i].record()
        if not local_copy:
            a2a_output = all_to_all_gemm_kernel.post_attn_a2a(
                input,
                seq_lens_cpu=seq_lens_cpu,
                all_to_all_option=option,
                num_comm_sm=num_comm_sm,
            )
        else:
            comm_buf = all_to_all_gemm_kernel.post_attn_a2a_no_cpy(
                input,
                seq_lens_cpu=seq_lens_cpu,
                all_to_all_option=option,
                num_comm_sm=num_comm_sm,
                comm_buf_idx=0,
            )
            all_to_all_gemm_kernel.sp_group_barrier_all()
            a2a_output = torch.empty(comm_buf.shape, dtype=comm_buf.dtype, device=comm_buf.device)
            a2a_output.copy_(comm_buf)
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
        name=f"flux  #{TP_GROUP.rank()}",
        a2a_output=a2a_output,
        total_ms=comm_time_ms,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bs", type=int)
    parser.add_argument("seq_len", type=int)
    parser.add_argument("nh", type=int)
    parser.add_argument("hd", type=int)
    parser.add_argument("--num_comm_sm", type=int, required=True, help="num sm for a2a")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument(
        "--fuse_sync", default=False, action="store_true", help="fuse sync into all2all kernel"
    )
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run once to verify correctness",
    )
    parser.add_argument(
        "--a2a_only", default=False, action="store_true", help="whether have transpose"
    )
    parser.add_argument("--dp", default=False, action="store_true", help="dp per rank")
    parser.add_argument("--sp_size", default=0, type=int, help="sp size")
    parser.add_argument(
        "--local-copy",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If local-copy is true, the user needs to copy output from comm buffer to user buffer",
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

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}

if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    args = parse_args()

    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()

    dtype = DTYPE_MAP[args.dtype]

    if dtype not in [torch.bfloat16]:
        raise NotImplementedError("A2A Gemm only support BF16.")

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

    assert args.nh % sp_group.size() == 0
    assert args.seq_len % sp_group.size() == 0

    local_nh = args.nh // sp_group.size()
    local_seq_len = args.seq_len // sp_group.size()
    # input: [bs, local_nh, seq_len, hd] for a2a_transpose, [bs, seq_len, local_nh, hd] for a2a_only
    if args.dp and not args.a2a_only:
        raise NotImplementedError("dp mode only support for a2a only")

    if not args.dp:
        seq_lens_cpu = None
        total_seq_len = args.seq_len
    else:
        seq_lens_list = list(
            np.random.randint(max(1, local_seq_len - 32), local_seq_len, size=(sp_group.size(),))
        )
        total_seq_len = sum(seq_lens_list)
        seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)
        if sp_group.rank() == 0:
            print(f"sp_group id = {RANK // sp_group.size()}, seq_lens_list = {seq_lens_list}")
    if not args.a2a_only:
        input_shape = [args.bs, local_nh, total_seq_len, args.hd]
    else:
        input_shape = [args.bs, total_seq_len, local_nh, args.hd]

    input = (-2 * torch.rand(input_shape, dtype=dtype).cuda() + 1) * (sp_group.rank() + 1)

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
            args.warmup,
            args.iters,
            args.a2a_only,
            args.dp,
            seq_lens_cpu=seq_lens_cpu,
        )
        perf_res_flux = perf_flux(
            sp_group,
            input,
            args.warmup,
            args.iters,
            args.num_comm_sm,
            args.fuse_sync,
            args.a2a_only,
            args.dp,
            args.local_copy,
            seq_lens_cpu=seq_lens_cpu,
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

    torch_output = perf_res_torch.a2a_output
    flux_output = perf_res_flux.a2a_output.reshape(torch_output.shape)
    torch.distributed.barrier()
    if flux.bitwise_check(torch_output, flux_output):
        print("✅  torch vs flux bitwise match")
    else:
        print("❌  torch vs flux not bitwise match")

    atol, rtol = 0.0, 0.0
    try:
        flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
    except Exception as e:
        torch.save(flux_output, f"flux_{TP_GROUP.rank()}.pt")
        torch.save(torch_output, f"torch_{TP_GROUP.rank()}.pt")
        print("❌ flux check failed")
        raise e
    else:
        print("✅ flux check passed")

    TP_GROUP.barrier()
    torch.cuda.synchronize()
