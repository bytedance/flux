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
import sys
from functools import partial

import torch
import torch.distributed

import flux
from flux.testing import DTYPE_MAP, initialize_distributed
from flux.triton.gemm_rs import GemmRSTritonNVLink as GemmRSTriton

print = partial(print, file=sys.stderr)


class GemmRSBaseline(torch.nn.Module):
    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        weight: torch.Tensor,
        max_m: int = 8192,
        n: int = 8192,
        k: int = 8192,
    ):
        self.pg = pg
        self.rank: int = pg.rank()
        self.world_size: int = pg.size()
        self.weight: torch.Tensor = weight
        self.n: int = n
        self.max_m: int = max_m
        self.k: int = k
        self.dtype: torch.dtype = weight.dtype

    def forward(self, x: torch.Tensor):
        M, local_K = x.shape
        assert M % self.world_size == 0
        assert local_K * self.world_size == self.k
        assert x.device == self.weight.device
        assert x.dtype == self.dtype
        output = torch.matmul(x, self.weight)
        rs_output = torch.empty((M // self.world_size, self.n), dtype=output.dtype, device=x.device)
        torch.distributed.reduce_scatter_tensor(rs_output, output, group=self.pg)
        return rs_output


def torch_allclose(x, y, rtol, atol):
    if not torch.allclose(x, y, rtol=rtol, atol=atol):
        print(f"shape of x: {x.shape}")
        print(f"shape of y: {y.shape}")

        print("x:", file=sys.stderr)
        print(x, file=sys.stderr)
        print("y:", file=sys.stderr)
        print(y, file=sys.stderr)
        print("x-y", x - y, file=sys.stderr)
        diff_loc = torch.isclose(x, y, rtol=rtol, atol=atol) == False
        print("x diff:", file=sys.stderr)
        print(x[diff_loc], file=sys.stderr)
        print("y diff:", file=sys.stderr)
        print(y[diff_loc], file=sys.stderr)
        num_diff = torch.sum(diff_loc)
        diff_rate = num_diff / (y.shape[0] * y.shape[1])
        print(
            f"diff count: {num_diff} ({diff_rate*100:.3f}%), {list(y.shape)}",
            file=sys.stderr,
        )
        max_diff = torch.max(torch.abs(x - y))
        rtol_abs = rtol * torch.min(torch.abs(y))
        print(f"diff max: {max_diff}, atol: {atol}, rtol_abs: {rtol_abs}", file=sys.stderr)
        diff_indices = (diff_loc == True).nonzero(as_tuple=False)
        print(f"diff locations:\n{diff_indices}", file=sys.stderr)
        print(
            "--------------------------------------------------------------\n",
            file=sys.stderr,
        )
        raise RuntimeError
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="float16", type=str, help="data type")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--dump_ptx", action="store_true")

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

    return parser.parse_args()


def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for n in range(iters + warmup_iters):
        if n == warmup_iters:
            start_event.record()
        output = func()
    stop_event.record()
    start_event.wait()
    stop_event.wait()
    torch.cuda.current_stream().synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters


if __name__ == "__main__":
    args = parse_args()
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    dtype = DTYPE_MAP[args.dtype]

    assert args.M % WORLD_SIZE == 0
    assert args.N % WORLD_SIZE == 0
    assert args.K % WORLD_SIZE == 0
    local_K = args.K // WORLD_SIZE

    input = (-2 * torch.rand((args.M, local_K), dtype=dtype).cuda() + 1) / 100 * (RANK + 1)
    weight = (-2 * torch.rand((args.N, local_K), dtype=dtype).cuda() + 1) / 100 * (RANK + 1)
    input_scale = None
    weight_scale = None

    bias = (
        torch.rand((args.M, args.N), dtype=dtype).cuda() / 10 * (RANK + 1)
        if args.has_bias
        else None
    )
    ctx = flux.get_torch_prof_ctx(args.profile)

    baseline_op = GemmRSBaseline(TP_GROUP, weight.t(), args.M, args.N, args.K)
    triton_op = GemmRSTriton(TP_GROUP, weight.t(), args.M, args.N, args.K, args.fuse, args.dump_ptx)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    run_baseline = partial(GemmRSBaseline.forward, baseline_op, input)
    run_triton = partial(GemmRSTriton.forward, triton_op, input)
    with ctx:
        torch_output, torch_duration_ms = perf_func(run_baseline, args.iters, args.warmup)
        triton_output, flux_duration_ms = perf_func(run_triton, args.iters, args.warmup)
        total_flops = args.M * local_K * args.N * 2
        torch_flops = total_flops / torch_duration_ms * 1000 / 1e12
        flux_flops = total_flops / flux_duration_ms * 1000 / 1e12

        print(f"torch: {torch_duration_ms:0.3f} ms {torch_flops:0.3f} TFLOPS")
        print(f"triton: {flux_duration_ms:0.3f} ms {flux_flops:0.3f} TFLOPS")

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{RANK}.json.gz")

    rtol = 1e-2
    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            try:
                if torch_allclose(triton_output, torch_output, rtol=rtol, atol=1e-2):
                    print("✅ Triton and Torch match")
                else:
                    print("❌ Triton and Torch differ")
            except RuntimeError as r:
                print(TP_GROUP.rank())
                print(r)
