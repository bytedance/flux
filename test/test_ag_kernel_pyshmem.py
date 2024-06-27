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

# usage: torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 --rdzv_id=none --master_addr=127.0.0.1 --master_port=23456 test/test_ag_kernel.py 4096 49152 12288
import argparse
import os
import sys
import time
import torch
import numpy as np
import datetime
import torch.distributed
from contextlib import nullcontext
import flux


RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

torch.cuda.set_device(LOCAL_RANK)
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=2)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
np.random.seed(3 + RANK)
# torch.set_printoptions(profile="full") ## debug


torch.distributed.init_process_group(
    backend="nccl", world_size=WORLD_SIZE, rank=RANK, timeout=datetime.timedelta(seconds=1800)
)
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")


class PerfResult:
    def __init__(
        self,
        name: str,
        output: torch.Tensor,
        total_ms: float,
        time1: str,
        gemm_time_ms: float,
        time2: str,
        comm_time_ms: float,
    ) -> None:
        self.name = name
        self.output = output
        self.total_ms = total_ms
        self.time1 = time1
        self.time2 = time2
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms

    def __repr__(self) -> str:
        return f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms, {self.time2} {self.comm_time_ms:.3f} ms"


@torch.no_grad()
def perf_torch(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, warmup: int, iters: int
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()

    torch.distributed.barrier()
    # All gather input tensors from all gpus
    full_input = torch.zeros(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)

    torch.distributed.barrier()
    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    allgather_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
        allgather_end_events[i].record()
        output = torch.matmul(full_input, weight.t())
        if bias is not None:
            output += bias
        end_events[i].record()

    comm_times = []  # all gather
    gemm_times = []  # gemm
    for i in range(total_iters):
        allgather_end_events[i].synchronize()
        end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(allgather_end_events[i]) / 1000)
            gemm_times.append(allgather_end_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        output=output,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    warmup: int,
    iters: int,
    transpose_weight: bool = True,
    local_copy: bool = False,
    gather_output: bool = False,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()
    K = input.size(1)

    if transpose_weight:
        w = weight.t().contiguous()
        N = w.size(1)

    else:
        w = weight
        N = w.size(0)

    torch.distributed.barrier()
    full_input = torch.zeros(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
    torch.distributed.barrier()

    gemm_only_op = flux.GemmOnly(w.dtype, transpose_weight=transpose_weight)
    all_gather_gemm_kernel = flux.AGKernel(
        TP_GROUP,
        NNODES,
        M,
        N,
        K,
        input.dtype,
        transpose_weight=transpose_weight,
        local_copy=local_copy,
    )

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_only_output_buf = torch.empty(
        [M, N], dtype=input.dtype, device=input.device, requires_grad=False
    )

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        _ = gemm_only_op.forward(full_input, w, bias=bias, output_buf=gemm_only_output_buf)
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
        all_gather_gemm_kernel.reset_signals()
        if local_copy:
            all_gather_gemm_kernel.copy_local(input)
        start_events[i].record()
        output = all_gather_gemm_kernel.forward(input, w, bias=bias)
        gathered = all_gather_gemm_kernel.gather() if gather_output else None
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    ag_gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            ag_gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    ag_gemm_time = sum(ag_gemm_times)

    ag_gemm_time_ms = ag_gemm_time / iters * 1000
    gemm_time_ms = gemm_time / iters * 1000
    comm_time_ms = (ag_gemm_time - gemm_time) / iters * 1000

    return PerfResult(
        name=f"flux  #{TP_GROUP.rank()}",
        output=output,
        total_ms=ag_gemm_time_ms,
        time1="gemm",
        gemm_time_ms=gemm_time_ms,
        time2="comm",
        comm_time_ms=comm_time_ms,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument(
        "--local_copy", default=True, action="store_true", help="perform local copy"
    )
    parser.add_argument(
        "--gather_output", default=False, action="store_true", help="output gather results"
    )
    parser.add_argument(
        "--transpose_weight", dest="transpose_weight", action="store_true", help="transpose weight"
    )
    parser.add_argument("--no_transpose_weight", dest="transpose_weight", action="store_false")
    parser.set_defaults(transpose_weight=True)
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    return parser.parse_args()


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}

if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    args = parse_args()
    print("before flux_shm initialization")
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()
    print("after flux_shm initialization")

    dtype = DTYPE_MAP[args.dtype]
    assert args.M % TP_GROUP.size() == 0
    assert args.N % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_M = args.M // TP_GROUP.size()
    local_N = args.N // TP_GROUP.size()

    # input: [M, K], weight: [N, K]
    input = torch.rand((local_M, args.K), dtype=dtype).cuda() / 100 * ((TP_GROUP.rank() + 1) ** 2)
    weight = torch.rand((local_N, args.K), dtype=dtype).cuda() / 100 * ((TP_GROUP.rank() + 1) ** 2)

    bias = None
    if args.has_bias:
        bias = torch.rand((args.M, local_N), dtype=dtype).cuda() / 10 * (TP_GROUP.rank() + 1)

    ctx = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        )
        if args.profile
        else nullcontext()
    )

    with ctx:
        perf_res_torch = perf_torch(input, weight, bias, args.warmup, args.iters)
        perf_res_flux = perf_flux(
            input,
            weight,
            bias,
            args.warmup,
            args.iters,
            args.transpose_weight,
            args.local_copy,
            args.gather_output,
        )

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    if TP_GROUP.rank() == 0:
        print(
            f"SOL time for GEMM(M={args.M},N={args.N},K={args.K},TP={TP_GROUP.size()}):"
            f" {flux.estimate_gemm_sol_time_ms(local_M, args.N, args.K):.3f}ms"
        )

    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            print(perf_res_torch)
        torch.distributed.barrier()
    torch.distributed.barrier()
    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            print(perf_res_flux)
        torch.distributed.barrier()
    torch.distributed.barrier()

    flux_output = perf_res_flux.output
    torch_output = perf_res_torch.output
    flux.torch_allclose(flux_output, torch_output, atol=1e-02, rtol=1e-02)
