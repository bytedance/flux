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
from functools import partial
from typing import Any, List, Optional

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import DTYPE_MAP, RING_MODE_MAP, MoeAgScatterWithTorch, MoeMlp1Ctx
from flux.testing.perf_db_helper import log_perf, set_global_args, should_log_to_rds
from flux.testing.utils import all_gather_into_tensor_with_fp8

try:
    from flux.triton.moe_ag_scatter import MoeAgScatterOp
except Exception as e:
    pass

DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
EP_GROUP = None
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
print = partial(print, flush=True)


def init_ep_group(ep_size: int):
    assert DIST_ENV.WORLD_SIZE % ep_size == 0, f"{DIST_ENV.WORLD_SIZE} % {ep_size} != 0"
    global EP_GROUP
    assert EP_GROUP is None, "EP_GROUP already initialized"

    assert TP_GROUP.size() % ep_size == 0, f"{TP_GROUP.size()} % {ep_size} != 0"
    ffn_tp_size = TP_GROUP.size() // ep_size

    temp_groups = []
    for i in range(ffn_tp_size):
        ranks = list(range(i, DIST_ENV.WORLD_SIZE, ffn_tp_size))
        temp_groups.append(ranks)

    ep_groups = []
    for group in temp_groups:
        for i in range(0, len(group), ep_size):
            ep_groups.append(group[i : i + ep_size])

    for ranks in ep_groups:
        group = DIST_ENV.new_group(ranks)
        if DIST_ENV.RANK in ranks:
            EP_GROUP = group


class PerfResult:
    def __init__(
        self,
        name: str,
        outputs: List[torch.Tensor],
        gathered_input: torch.Tensor,
        gemm_time_ms: float,
        scatter_time_ms: float,
        comm_time_ms: float,
    ) -> None:
        self.name = name
        self.outputs = outputs
        self.gathered_input = gathered_input
        self.gemm_time_ms = gemm_time_ms
        self.scatter_time_ms = scatter_time_ms
        self.comm_time_ms = comm_time_ms
        self.total_ms = self.gemm_time_ms + self.scatter_time_ms + self.comm_time_ms

    def __repr__(self) -> str:
        return (
            f"{self.name}: gemm {self.gemm_time_ms:.3f} ms"
            f", scatter {self.scatter_time_ms:.3f} ms"
            f", comm {self.comm_time_ms:.3f} ms"
        )


def take_first_or_none(x: Optional[List[Any]]):
    return x[0] if x is not None else None


@torch.no_grad()
def perf_lego(ctx: MoeMlp1Ctx, warmup_iters: int, iters: int, gather_input: bool = True):

    import lego_ops

    lego_ops.load_ft_torch()
    token_cnt = ctx.splits_gpu.to("cpu")[
        ctx.nexperts_ep * ctx.ep_rank : ctx.nexperts_ep * ctx.ep_rank + ctx.nexperts_ep
    ]

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    comm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    scatter_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    ctx.clear_outputs()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    inp_scal_inv = torch.ones(1, dtype=torch.float32, device=torch.cuda.current_device())
    weight_fp8 = ctx.weights[0]
    weight_scal_inv = ctx.output_scale[0]
    _2X_ACC_FPROP = not ctx.fast_accum
    output_base = ctx.outputs[0]
    row_start = int(ctx.splits_cpu[: ctx.nexperts_ep * ctx.ep_rank].sum())
    row_end = int(ctx.splits_cpu[: ctx.nexperts_ep * (ctx.ep_rank + 1)].sum())
    for i in range(total_iters):
        ctx.clear_outputs()
        start_events[i].record()
        all_gather_into_tensor_with_fp8(ctx.inputs, ctx.inputs_shard, group=TP_GROUP)
        comm_end_events[i].record()
        MoeAgScatterWithTorch.scatter_impl(ctx)
        scatter_end_events[i].record()
        input_fp8 = ctx.scatter_inputs[row_start:row_end]
        output = torch.ops.FP8.moe_linear_fp8_fwd_v2(
            input_fp8,
            weight_fp8,
            None,  # bias
            token_cnt,
            inp_scal_inv,
            weight_scal_inv,
            None,  # output scale
            not _2X_ACC_FPROP,
            output_base,
        )
        gemm_end_events[i].record()

    comm_times = []
    scatter_times = []
    gemm_times = []
    for i in range(total_iters):
        comm_end_events[i].synchronize()
        scatter_end_events[i].synchronize()
        gemm_end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(comm_end_events[i]))
            scatter_times.append(comm_end_events[i].elapsed_time(scatter_end_events[i]))
            gemm_times.append(scatter_end_events[i].elapsed_time(gemm_end_events[i]))
    comm_time = sum(comm_times) / iters
    scatter_time = sum(scatter_times) / iters
    gemm_time = sum(gemm_times) / iters

    return PerfResult(
        name=f"lego   #{TP_GROUP.rank()}",
        outputs=[output],
        gathered_input=flux.testing.clone_with_fp8(ctx.inputs),
        gemm_time_ms=gemm_time,
        scatter_time_ms=scatter_time,
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_torch(ctx: MoeMlp1Ctx, warmup_iters: int, iters: int, gather_input: bool = True):
    gemm_only_op = flux.GemmOnly(
        ctx.inputs.dtype,
        ctx.outputs[0].dtype,
        use_fp8_gemm=flux.util.get_arch() < 90 and flux.is_fp8_dtype(ctx.inputs.dtype),
    )

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    comm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    scatter_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    ctx.clear_outputs()
    torch.distributed.barrier()
    torch.cuda.synchronize()

    for i in range(total_iters):
        start_events[i].record()
        MoeAgScatterWithTorch.comm_impl(ctx, TP_GROUP)
        comm_end_events[i].record()
        MoeAgScatterWithTorch.scatter_impl(ctx)
        scatter_end_events[i].record()
        MoeAgScatterWithTorch.gemm_impl(ctx, gemm_only_op)
        gemm_end_events[i].record()
    comm_times = []
    scatter_times = []
    gemm_times = []
    for i in range(total_iters):
        comm_end_events[i].synchronize()
        scatter_end_events[i].synchronize()
        gemm_end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(comm_end_events[i]))
            scatter_times.append(comm_end_events[i].elapsed_time(scatter_end_events[i]))
            gemm_times.append(scatter_end_events[i].elapsed_time(gemm_end_events[i]))
    comm_time = sum(comm_times) / iters
    scatter_time = sum(scatter_times) / iters
    gemm_time = sum(gemm_times) / iters

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        outputs=ctx.get_outputs_clone(),
        gathered_input=flux.testing.clone_with_fp8(ctx.inputs),
        gemm_time_ms=gemm_time,
        scatter_time_ms=scatter_time,
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_triton(
    ctx: MoeMlp1Ctx,
    warmup_iters: int,
    iters: int,
    gather_input: bool = True,
    ag_option=flux.AllGatherOption(),
):
    op = MoeAgScatterOp(
        tp_group=TP_GROUP,
        ep_group=EP_GROUP,
        max_ntokens=ctx.b * ctx.s,
        hidden=ctx.h,
        ffn_hidden=ctx.ffn_size,
        nexperts=ctx.nexperts,
        topk=ctx.topk,
        input_dtype=ctx.inputs_shard.dtype,
    )

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gathered_input = torch.empty_like(ctx.inputs) if gather_input else None
    ctx.clear_outputs()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    for i in range(total_iters):
        ctx.clear_outputs()
        start_events[i].record()
        op.forward(
            input=ctx.inputs_shard,
            weights=ctx.weights[0],
            bias=ctx.bias[0] if ctx.bias is not None else None,
            input_scale=ctx.input_scale[0] if ctx.input_scale is not None else None,
            weight_scale=ctx.weight_scale[0] if ctx.weight_scale is not None else None,
            output_scale=ctx.output_scale[0] if ctx.output_scale is not None else None,
            splits_gpu=ctx.splits_gpu,
            scatter_index=ctx.scatter_index,
            outputs_buf=ctx.outputs[0],
            fast_accum=ctx.fast_accum,
            gathered_input=gathered_input,
            ag_option=ag_option,
        )
        end_events[i].record()

    gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_times.append(start_events[i].elapsed_time(end_events[i]))

    gemm_time_ms = sum(gemm_times) / iters

    return PerfResult(
        name=f"triton #{TP_GROUP.rank()}",
        outputs=ctx.get_outputs_clone(),
        gathered_input=gathered_input,
        gemm_time_ms=gemm_time_ms,
        scatter_time_ms=0.0,
        comm_time_ms=0.0,
    )


@torch.no_grad()
def perf_flux(
    ctx: MoeMlp1Ctx,
    warmup_iters: int,
    iters: int,
    gather_input: bool = True,
    ag_option: flux.AllGatherOption = flux.AllGatherOption(),
):
    tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=DIST_ENV.NNODES, ep_group=EP_GROUP)
    moe_args = flux.MoeArguments(
        max_ntokens=ctx.b * ctx.s,
        hidden=ctx.h,
        ffn_hidden=ctx.ffn_size,
        nexperts=ctx.nexperts,
        topk=ctx.topk,
        input_dtype=ctx.inputs_shard.dtype,
        output_dtype=ctx.outputs[0].dtype,
    )

    extra_args = {}
    if flux.util.get_arch() >= 90:
        op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
    else:
        op = flux.GemmGroupedV2AGScatterOp(tp_env=tp_env, moe_args=moe_args)
        extra_args = {
            "ag_option": ag_option,
        }
        if ctx.weight_groups == 1:
            extra_args.update(
                {
                    "bias": take_first_or_none(ctx.bias),
                    "input_scale": take_first_or_none(ctx.input_scale),
                    "weight_scale": take_first_or_none(ctx.weight_scale),
                }
            )
        else:
            extra_args.update(
                {
                    "bias": ctx.bias,
                    "input_scale": ctx.input_scale,
                    "weight_scale": ctx.weight_scale,
                }
            )

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    ctx.clear_outputs()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    gathered_input = torch.empty_like(ctx.inputs) if gather_input else None
    for i in range(total_iters):
        ctx.clear_outputs()
        op.clear_buffers()
        start_events[i].record()
        if ctx.weight_groups == 1:
            op.forward(
                inputs_shard=ctx.inputs_shard,
                weights=ctx.weights[0],
                splits_gpu=ctx.splits_gpu,
                scatter_index=ctx.scatter_index,
                output_scale=take_first_or_none(ctx.output_scale),
                outputs_buf=take_first_or_none(ctx.outputs),
                fast_accum=ctx.fast_accum,
                sm_margin=args.sm_margin,
                allgather_output=gathered_input,
                **extra_args,
            )
        else:
            op.forward_multiple_weights(
                inputs_shard=ctx.inputs_shard,
                weights=ctx.weights,
                splits_gpu=ctx.splits_gpu,
                scatter_index=ctx.scatter_index,
                output_scale=ctx.output_scale,
                outputs_buf=ctx.outputs,
                fast_accum=ctx.fast_accum,
                sm_margin=args.sm_margin,
                allgather_output=gathered_input,
                **extra_args,
            )
        end_events[i].record()

    gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_times.append(start_events[i].elapsed_time(end_events[i]))

    gemm_time_ms = sum(gemm_times) / iters

    return PerfResult(
        name=f"flux #{TP_GROUP.rank()}",
        outputs=ctx.get_outputs_clone(),
        gathered_input=gathered_input,
        gemm_time_ms=gemm_time_ms,
        scatter_time_ms=0.0,
        comm_time_ms=0.0,
    )


@torch.no_grad()
def tune_flux(ctx: MoeMlp1Ctx) -> flux.ProfilingContext:
    name = f"config_ag_scatter_sm{flux.get_arch()}"
    prof_ctx = flux.ProfilingContext(name)
    tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=DIST_ENV.NNODES, ep_group=EP_GROUP)
    moe_args = flux.MoeArguments(
        max_ntokens=ctx.b * ctx.s,
        hidden=ctx.h,
        ffn_hidden=ctx.ffn_size,
        nexperts=ctx.nexperts,
        topk=ctx.topk,
        input_dtype=ctx.inputs_shard.dtype,
        output_dtype=ctx.outputs[0].dtype,
    )

    if flux.util.get_arch() >= 90:
        op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
    else:
        op = flux.GemmGroupedV2AGScatterOp(tp_env=tp_env, moe_args=moe_args)

    op.profiling(
        inputs_shard=ctx.inputs_shard,
        weights=ctx.weights,
        splits_gpu=ctx.splits_gpu,
        scatter_index=ctx.scatter_index,
        output_scale=ctx.output_scale,
        outputs_buf=ctx.outputs,
        fast_accum=ctx.fast_accum,
        prof_ctx=prof_ctx,
    )
    torch.cuda.synchronize()
    return prof_ctx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist",
        type=str,
        default="random",
        choices=["uniform", "random_uniform", "random", "random_with_first_k_experts"],
    )
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S", type=int, default=4096)
    parser.add_argument("--H", type=int, default=8192)
    parser.add_argument("--ffn_hidden_size", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--G", type=int, default=32)
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--warmup_iters", default=10, type=int, help="warmup iterations")
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument(
        "--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys())
    )
    parser.add_argument("-E", "--E", default=1, type=int, help="ep size")
    parser.add_argument("--weight_groups", default=1, type=int, help="num of weight groups")
    parser.add_argument(
        "--fast_accum", default=False, action="store_true", help="fp8 use fast accum"
    )
    parser.add_argument("--triton", default=False, action="store_true", help="use triton")
    parser.add_argument("--lego", default=False, action="store_true", help="use triton")
    parser.add_argument(
        "--triton-only",
        default=False,
        action="store_true",
        help="run triton only. don't rn flux. maybe flux is not implemented yet",
    )
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument(
        "--drop_token",
        default=False,
        action="store_true",
        help="if True, splits will have an additional item, tokens in whitch are droped",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
    parser.add_argument(
        "--bias", action=argparse.BooleanOptionalAction, default=False, help="whether to add bias"
    )
    parser.add_argument("--tune", default=False, action="store_true", help="find best GemmHParams")
    parser.add_argument(
        "--gather_input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="gather input",
    )
    parser.add_argument(
        "--stable_index",
        default=False,  # use harder cases.
        action="store_true",
        help="use sorted gather_index for each expert",
    )
    parser.add_argument(
        "--ring_mode",
        default="auto",
        choices=["auto", "all2all", "ring1d", "ring2d"],
        help="ring mode. auto for auto detect",
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
    return parser.parse_args()


OUT_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.bfloat16,
    "float8_e5m2": torch.bfloat16,
    "s8": torch.bfloat16,
}

if __name__ == "__main__":
    args = parse_args()
    init_ep_group(args.E)

    print("before flux_shm initialization")
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()
    print("after flux_shm initialization")

    input_dtype = DTYPE_MAP[args.dtype]
    is_fp8 = flux.util.is_fp8_dtype(input_dtype)
    is_s8 = torch.int8 == input_dtype
    output_dtype = torch.bfloat16 if is_fp8 or is_s8 else input_dtype
    generator = torch.Generator(device="cuda")
    generator.manual_seed(12345)  # use the same random generator to generate the same distribution
    moe_ctx = MoeMlp1Ctx(
        TP_GROUP,
        EP_GROUP,
        b=args.B,
        s=args.S,
        h=args.H,
        ffn_size=args.ffn_hidden_size,
        nexperts=args.G,
        topk=args.topk,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        dist=args.dist,
        fast_accum=args.fast_accum,
        weight_groups=args.weight_groups,
        drop_token=args.drop_token,
        debug=args.debug,
        generator=generator,
        stable=args.stable_index,
    )

    if TP_GROUP.rank() == 0:
        print(
            f"Splits:{moe_ctx.splits_cpu.tolist()}, Shape:{moe_ctx.splits_cpu.shape}, Sum:{sum(moe_ctx.splits_cpu.tolist())}"
        )

    if args.tune:
        prof_ctx = tune_flux(moe_ctx)

        if DIST_ENV.RANK == 0:
            print("====== Profiling Results =======")
            print("\n".join(prof_ctx.get_all_prof_results()))
            print("====== Generated Config Code =======")
            print(prof_ctx.get_code())

        flux.load_tuning_record(prof_ctx.get_latest_record())

    if args.triton_only:
        if not args.triton:
            print("WARNING: force set --triton with --triton-only set.")
            args.triton = True

    if args.triton:
        assert args.weight_groups == 1, f"triton implementation does not support multiple group yet"

    ag_option = flux.AllGatherOption()
    ag_option.use_cuda_core_local = args.use_cuda_core_local
    ag_option.use_cuda_core_ag = args.use_cuda_core_ag
    ag_option.mode = RING_MODE_MAP[args.ring_mode]

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    with flux.group_profile(
        name="moe_ag_scatter_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile,
        group=TP_GROUP,
    ):
        if args.lego:
            perf_result_lego = perf_lego(moe_ctx, args.warmup_iters, args.iters, args.gather_input)
        if not args.triton_only:
            perf_result_flux = perf_flux(
                moe_ctx, args.warmup_iters, args.iters, args.gather_input, ag_option
            )
        perf_result_torch = perf_torch(moe_ctx, args.warmup_iters, args.iters, args.gather_input)
        if args.triton:
            perf_result_triton = perf_triton(
                moe_ctx, args.warmup_iters, args.iters, args.gather_input, ag_option=ag_option
            )

    if TP_GROUP.rank() == 0:
        flux.testing.print_grouped_gemm_sol_time_ms(
            moe_ctx.ntokens * moe_ctx.topk * args.weight_groups,
            moe_ctx.ffn_size_shard,
            moe_ctx.h,
            args.G // args.E,  # E
            input_dtype=input_dtype,
        )
    if should_log_to_rds():
        set_global_args("moe_ag_scatter", args)
    flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_torch))
    if not args.triton_only:
        flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_flux))
    if args.triton:
        flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_triton))
    if args.lego:
        flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_lego))

    if input_dtype == torch.float16:
        atol, rtol = 1e-2, 1e-3
    elif input_dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1.5e-2
    elif input_dtype == torch.int8:
        atol, rtol = 1e-9, 1e-9  # bitwise match
    elif is_fp8:
        if args.fast_accum:
            atol, rtol = 1e-2, 3 / 2**7
        else:
            atol, rtol = 1e-2, 1.5e-2
    else:
        raise ValueError(f"Unsupported dtype {input_dtype}")

    def check_result(perf_out_x, perf_out_y, name_x: str, name_y: str):
        print(f"Checking RANK #{TP_GROUP.rank()}...")
        if args.gather_input:
            assert flux.testing.bitwise_eq(perf_out_x.gathered_input, perf_out_y.gathered_input)
        for x, y in zip(perf_out_x.outputs, perf_out_y.outputs):
            print("output shape", x.size())
            if flux.testing.bitwise_eq(x, y):
                print(f"✅ {name_x} and torch bitwise match")
            else:
                print(f"❌ {name_x} and torch not bitwise match")
            try:
                flux.torch_allclose(x, y, atol=atol, rtol=rtol)
            except Exception as e:
                torch.save(x, f"{name_x}_{TP_GROUP.rank()}.pt")
                torch.save(y, f"{name_y}_{TP_GROUP.rank()}.pt")
                torch.save(moe_ctx, f"moe_ctx_{TP_GROUP.rank()}.pt")
                print(f"❌ {name_x} check failed")
                raise e
            else:
                print(f"✅ {name_x} check passed")

    if args.lego:
        flux.exec_in_rank_order(
            TP_GROUP, lambda: check_result(perf_result_lego, perf_result_torch, "lego", "torch")
        )
    if not args.triton_only:
        flux.exec_in_rank_order(
            TP_GROUP, lambda: check_result(perf_result_flux, perf_result_torch, "flux", "torch")
        )
    if args.triton:
        flux.exec_in_rank_order(
            TP_GROUP, lambda: check_result(perf_result_triton, perf_result_torch, "triton", "torch")
        )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
