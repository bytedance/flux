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
import os
import time
from functools import partial
from typing import Optional

import torch
import torch.distributed

import flux
from flux.testing import (
    DTYPE_MAP,
    RING_MODE_MAP,
    all_gather_into_tensor_with_fp8,
    generate_data,
    initialize_distributed,
    zeros_with_fp8,
    matmul_int8,
)
import flux.testing
from flux.testing.perf_db_helper import should_log_to_rds, set_global_args, log_perf
from flux.util import bench_func, is_fp8_dtype

try:
    from flux.triton.ag_gemm import AgGemmTriton
except Exception as e:
    print("triton module import failed. skip...")

print = partial(print, flush=True)


class PerfResult:
    def __init__(
        self,
        name: str,
        output: torch.Tensor,
        gathered_output: torch.Tensor,
        total_ms: float,
        time1: str,
        gemm_time_ms: float,
        time2: str,
        comm_time_ms: float,
        time3: str = "gemm_only",
        gemm_only_time_ms: float = 0,
    ) -> None:
        self.name = name
        self.output = output
        self.gathered_output = gathered_output
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


@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    is_fp8: bool,
    is_s8_dequant: bool,
    warmup: int,
    iters: int,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()

    torch.distributed.barrier()
    # All gather input tensors from all gpus
    full_input = zeros_with_fp8(
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

    alpha_scale = 1.0
    if is_fp8:
        alpha_scale = input_scale * weight_scale
        input = input.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        full_input = full_input.to(torch.bfloat16)

    if is_s8_dequant:
        assert input_scale is not None
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)
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
        if is_s8_dequant:
            accum = matmul_int8(full_input, weight.t()).to(torch.float32)
            output = full_input_scale * weight_scale * accum
        else:
            output = alpha_scale * torch.matmul(full_input, weight.t())

        if is_fp8 or is_s8_dequant:
            output = output.to(torch.bfloat16)
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
        gathered_output=full_input,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_flux_no_overlap(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    transpose_weight: bool = True,
    gather_input: bool = True,  # not used. always as true
    warmup: int = 5,
    iters: int = 10,
    fast_acc: bool = False,
):
    input_dtype = input.dtype
    is_fp8 = is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input.dtype
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()
    K = input.size(1)

    if transpose_weight:
        w = weight.t().contiguous()
        N = w.size(1)
    else:
        w = weight
        N = w.size(0)

    full_input = zeros_with_fp8(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    full_input_scale = (
        torch.zeros((M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device())
        if is_s8_dequant
        else None
    )
    all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
    if is_s8_dequant:
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)

    ag_gemm_op = flux.AGKernel(
        TP_GROUP,
        NNODES,
        M,
        N,
        K,
        input_dtype,
        output_dtype=output_dtype,
    )

    gemm_only_output = torch.empty(
        [M, N], dtype=output_dtype, device=input.device, requires_grad=False
    )

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    allgather_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
        allgather_end_events[i].record()

        gemm_only_output = ag_gemm_op.gemm_only(
            full_input,
            w,
            bias=bias,
            input_scale=full_input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            transpose_weight=transpose_weight,
        )
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
        name=f"flux(no-overlap) #{TP_GROUP.rank()}",
        output=gemm_only_output,
        gathered_output=full_input,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    transpose_weight: bool = True,
    gather_input: bool = False,
    warmup: int = 5,
    iters: int = 10,
    fast_acc: bool = False,
    verify: bool = False,
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

    op = AgGemmTriton(TP_GROUP, weight.dtype, M, K, transpose_weight=transpose_weight)

    full_input = (
        zeros_with_fp8(
            (M, K),
            dtype=input.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        if gather_input
        else None
    )

    ag_option = flux.AllGatherOption()
    ag_option.mode = RING_MODE_MAP[args.ring_mode]

    (output, gathered_output), total_time_ms = bench_func(
        lambda: op.forward(
            input,
            w.t(),
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            gathered_input=full_input,
            fast_accum=fast_acc,
            ag_option=ag_option,
        ),
        iters,
        warmup,
    )
    return PerfResult(
        name=f"triton  #{TP_GROUP.rank()}",
        output=output,
        gathered_output=gathered_output,
        total_ms=total_time_ms,
        time1="gemm",
        gemm_time_ms=0,
        time2="comm",
        comm_time_ms=0,
    )


@torch.no_grad()
def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    transpose_weight: bool = True,
    gather_input: bool = False,
    ring_mode: Optional[flux.AGRingMode] = None,
    warmup: int = 5,
    iters: int = 10,
    fast_acc: bool = False,
    verify: bool = False,
    use_cuda_core_local: bool = False,
    use_cuda_core_ag: bool = False,
    use_pdl: bool = False,
):
    input_dtype = input.dtype
    is_fp8 = is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input_dtype
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
    full_input = zeros_with_fp8(
        (M, K),
        dtype=input_dtype,
        device=torch.cuda.current_device(),
    )

    full_input_scale = (
        torch.zeros((M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device())
        if is_s8_dequant
        else None
    )
    all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
    if is_s8_dequant:
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)

    use_fp8_gemm = True if is_fp8 else False
    gemm_only_op = flux.GemmOnly(
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        transpose_weight=transpose_weight,
        use_fp8_gemm=use_fp8_gemm,
    )
    gemm_only_output = torch.empty(
        [M, N], dtype=output_dtype, device=input.device, requires_grad=False
    )

    ag_gemm_output = torch.empty([M, N], dtype=output_dtype, device=input.device)
    ag_option = flux.AllGatherOption()
    ag_option.mode = ring_mode
    all_gather_gemm_kernel = flux.AGKernel(
        TP_GROUP,
        NNODES,
        M,
        N,
        K,
        input_dtype,
        output_dtype=output_dtype,
        use_pdl=use_pdl,
    )

    warmup_iters = warmup
    total_iters = warmup_iters + iters if not verify else 1
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    for i in range(total_iters):
        start_events[i].record()
        gemm_only_output = gemm_only_op.forward(
            full_input,
            w,
            bias=bias,
            output_buf=gemm_only_output,
            input_scale=input_scale if not is_s8_dequant else full_input_scale,
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

    full_input.zero_()
    time.sleep(1)

    torch.distributed.barrier()
    ag_option.use_cuda_core_local = use_cuda_core_local
    ag_option.use_cuda_core_ag = use_cuda_core_ag
    for i in range(total_iters):
        start_events[i].record()
        all_gather_gemm_kernel.forward(
            input,
            w,
            bias=bias,
            output=ag_gemm_output,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            gathered_input=full_input if gather_input else None,
            transpose_weight=transpose_weight,
            all_gather_option=ag_option,
        )
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    ag_gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            ag_gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    ag_gemm_time = sum(ag_gemm_times)

    ## signals are already set
    for i in range(total_iters):
        start_events[i].record()
        if not verify:
            _ = all_gather_gemm_kernel.gemm_only(
                full_input,
                w,
                bias=bias,
                input_scale=full_input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=fast_acc,
                transpose_weight=transpose_weight,
            )
        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    gemm_only_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_only_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)
    gemm_only_time = sum(gemm_only_times)

    ag_gemm_time_ms = ag_gemm_time / iters * 1000
    gemm_time_ms = gemm_time / iters * 1000
    comm_time_ms = (ag_gemm_time - gemm_time) / iters * 1000
    gemm_only_time_ms = gemm_only_time / iters * 1000

    is_bitwise_match = flux.bitwise_check(gemm_only_output, ag_gemm_output)
    if TP_GROUP.rank() == 0:
        print("is bitwise match: ", is_bitwise_match)

    return PerfResult(
        name=f"flux  #{TP_GROUP.rank()}",
        output=ag_gemm_output,
        gathered_output=full_input,
        total_ms=ag_gemm_time_ms,
        time1="gemm",
        gemm_time_ms=gemm_time_ms,
        time2="comm",
        comm_time_ms=comm_time_ms,
        time3="gemm_only",
        gemm_only_time_ms=gemm_only_time_ms,
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
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=True,
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument(
        "--fastacc",
        default=False,
        action="store_true",
        help="whether to use fast accumulation (FP8 Gemm only)",
    )
    parser.add_argument(
        "--ring_mode",
        default="auto",
        choices=["auto", "all2all", "ring1d", "ring2d"],
        help="ring mode. auto for auto detect",
    )
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run once to verify correctness",
    )

    parser.add_argument(
        "--use_cuda_core_local",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl local copy, auto select if not specified",
    )

    parser.add_argument(
        "--use_cuda_core_ag",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl all gather, auto select if not specified",
    )

    parser.add_argument(
        "--use_pdl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use Programmatic Dependent Launch",
    )

    parser.add_argument(
        "--triton",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run with triton kernels",
    )
    parser.add_argument(
        "--gather_input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="gather input",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
    return parser.parse_args()


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
    torch.int8: 0,
}

if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()

    input_dtype = DTYPE_MAP[args.dtype]
    is_fp8 = is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8

    if args.transpose_weight and (is_fp8 or is_s8_dequant):
        raise ValueError("FP8/S8 GEMM does not support transpose weight (RRR layout).")

    assert args.M % TP_GROUP.size() == 0
    assert args.N % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_M = args.M // TP_GROUP.size()
    local_N = args.N // TP_GROUP.size()

    scale = TP_GROUP.rank() + 1
    if is_s8_dequant:
        data_config = [
            ((local_M, args.K), input_dtype, (127, 0)),  # A
            ((local_N, args.K), input_dtype, (127, 0)),  # B
            None if not args.has_bias else ((1, local_N), torch.bfloat16, (scale, 0)),  # bias
            ((local_M, 1), torch.float32, (1, 0)),  # input_scale
            ((1, local_N), torch.float32, (1, 0)),  # weight_scale
        ]
    elif is_fp8:
        data_config = [
            ((local_M, args.K), input_dtype, (0.01 * scale, 0)),  # A
            ((local_N, args.K), input_dtype, (0.01 * scale, 0)),  # B
            (  # bias
                None if not args.has_bias else ((1, local_N), torch.bfloat16, (0.1 * scale, 0))
            ),
            ((1, 1), torch.float32, (1, 0)),  # input_scale
            ((1, 1), torch.float32, (1, 0)),  # weight_scale
        ]
    else:
        data_config = [
            ((local_M, args.K), input_dtype, (0.01 * scale, 0)),  # A
            ((local_N, args.K), input_dtype, (0.01 * scale, 0)),  # B
            (  # bias
                None if not args.has_bias else ((args.M, local_N), input_dtype, (0.1 * scale, 0))
            ),
            None,  # input_scale
            None,  # weight_scale
        ]

    generator = generate_data(data_config)
    input, weight, bias, input_scale, weight_scale = next(generator)

    if args.debug:
        input.zero_()
        input[:, 0].fill_(TP_GROUP.rank() + 1)
        weight.fill_(1)
        if input_scale is not None:
            input_scale.fill_(1)
            weight_scale.fill_(1)
        if bias is not None:
            bias.zero_()
    TP_GROUP.barrier()

    with flux.util.group_profile(
        name="ag_gemm_" + os.environ["TORCHELASTIC_RUN_ID"], do_prof=args.profile, group=TP_GROUP
    ):
        perf_res_torch = perf_torch(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            is_fp8,
            is_s8_dequant,
            args.warmup,
            args.iters,
        )
        perf_res_flux = perf_flux(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            args.transpose_weight,
            args.gather_input,
            RING_MODE_MAP[args.ring_mode],
            args.warmup,
            args.iters,
            args.fastacc,
            args.verify,
            args.use_cuda_core_local,
            args.use_cuda_core_ag,
            args.use_pdl,
        )

        perf_res_flux_no_overlap = perf_flux_no_overlap(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            args.transpose_weight,
            args.gather_input,  # not used,
            args.warmup,
            args.iters,
            args.fastacc,
        )

        if args.triton:
            perf_res_triton = perf_triton(
                input,
                weight,
                bias,
                input_scale,
                weight_scale,
                args.transpose_weight,
                args.gather_input,
                args.warmup,
                args.iters,
                args.fastacc,
                args.verify,
            )

    if TP_GROUP.rank() == 0:
        flux.testing.print_gemm_sol_time(local_M, args.N, args.K, input_dtype)

    if should_log_to_rds():
        set_global_args("ag_gemm", args)
    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            log_perf(perf_res_torch)
            log_perf(perf_res_flux)
            log_perf(perf_res_flux_no_overlap)
            if args.triton:
                log_perf(perf_res_triton)
        torch.distributed.barrier()

    torch_output = perf_res_torch.output
    flux_output = perf_res_flux.output
    torch_gathered_data = perf_res_torch.gathered_output
    flux_gathered_data = perf_res_flux.gathered_output
    torch.distributed.barrier()
    if flux.bitwise_check(torch_output, flux_output):
        print("✅  torch vs flux bitwise match")
    else:
        print("❌  torch vs flux not bitwise match")

    atol = THRESHOLD_MAP[input_dtype]
    rtol = THRESHOLD_MAP[input_dtype]
    if args.gather_input:
        try:
            flux.torch_allclose(flux_gathered_data, torch_gathered_data, atol=1e-9, rtol=1e-9)
        except Exception as e:
            torch.save(flux_gathered_data, f"flux_gathered_data_{TP_GROUP.rank()}.pt")
            torch.save(torch_gathered_data, f"torch_gathered_data_{TP_GROUP.rank()}.pt")
            print("❌ flux gathered data check failed")
            raise e
        else:
            print("✅ flux gathered data check passed")
    try:
        flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
    except Exception as e:
        torch.save(flux_output, f"flux_{TP_GROUP.rank()}.pt")
        torch.save(torch_output, f"torch_{TP_GROUP.rank()}.pt")
        print("❌ flux check failed")
        raise e
    else:
        print("✅ flux check passed")

    if args.triton:
        triton_output = perf_res_triton.output
        is_bitwise_match = flux.bitwise_check(torch_output, triton_output)
        print("torch vs triton bitwise match: ", is_bitwise_match)
        if args.gather_input:
            triton_gathered_data = perf_res_triton.gathered_output
            try:
                flux.torch_allclose(triton_gathered_data, flux_gathered_data, atol=1e-9, rtol=1e-9)
            except Exception as e:
                torch.save(triton_gathered_data, f"triton_gathered_data_{TP_GROUP.rank()}.pt")
                torch.save(torch_gathered_data, f"torch_gathered_data_{TP_GROUP.rank()}.pt")
                print("❌ triton gathered data check failed")
                raise e
            else:
                print("✅ triton gathered data check passed")
        try:
            flux.torch_allclose(triton_output, torch_output, atol=atol, rtol=rtol)
        except Exception as e:
            torch.save(triton_output, f"triton_{TP_GROUP.rank()}.pt")
            torch.save(torch_output, f"torch_{TP_GROUP.rank()}.pt")
            print("❌ triton check failed")
            raise e
        else:
            print("✅ triton check passed")

    TP_GROUP.barrier()
    torch.cuda.synchronize()
