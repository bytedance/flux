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
from typing import List

import torch
from copy_utils import copy_continous, copy_cudaMemcpyAsync, reduce_continous, set_cuda_p2p_access

import flux
from flux.testing import DTYPE_MAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", default=8192, type=int, help="")
    parser.add_argument("n", default=8192, type=int, help="")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--iters", default=10, type=int)
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument("--num_threads", type=int, default=1024)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--reduce", action="store_true", default=False)
    parser.add_argument(
        "--copy_jobs",
        default="0:cpu,cuda0",
        help=f"split by ';', each item in format <device-id>:<from>,<to>, from/to can be cpu or cuda{{0~N}}. for example, 0:cpu,cuda0;3:cuda0,cpu mean GPU0 copy from cpu to GPU0, GPU3 copy from GPU0 to CPU",
    )

    return parser.parse_args()


class CopyJob:
    def __init__(self, device_id, from_device, to_device) -> None:
        self.device_id = device_id
        self.from_device = from_device
        self.to_device = to_device

    def __repr__(self) -> str:
        return f"{self.device_id}: {self.from_device} -> {self.to_device}"


def _parse_copy_jobs(opt: str):
    def _format_device(device: str):
        if device.startswith("cuda"):
            device = f"cuda:{device[4:]}"
        if device.startswith("gpu"):
            device = f"cuda:{device[3:]}"
        torch.device(device)
        return device

    def _parse_job(job: str):
        device_id, from_to = job.split(":")
        device_id = int(device_id)
        from_device, to_device = from_to.split(",")
        return CopyJob(device_id, _format_device(from_device), _format_device(to_device))

    jobs = opt.split(";")
    return [_parse_job(x) for x in jobs]


class Bench:
    def __init__(self, job: CopyJob, shape, dtype) -> None:
        self.device_id = job.device_id
        with torch.cuda.device(self.device_id):
            self.src = self._create_tensor(shape, dtype, job.from_device)
            self.dst = self._create_tensor(shape, dtype, job.to_device)
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.stop_event = torch.cuda.Event(enable_timing=True)
            self.stream = torch.cuda.Stream(self.device_id)

    def _create_tensor(self, shape, dtype, device) -> torch.Tensor:
        extra_args = {"pin_memory": True} if device == "cpu" else {}
        return torch.zeros(shape, dtype=dtype, device=device, **extra_args)

    def sync(self):
        self.stream.synchronize()

    def start(self):
        self.start_event.record(self.stream)

    def stop(self):
        self.stop_event.record(self.stream)

    def duration(self):
        return self.start_event.elapsed_time(self.stop_event)

    def run(
        self,
        use_cudaMemcpy: bool = False,
        do_reduce: bool = False,
        num_blocks: int = 4,
        num_threads: int = 1024,
    ) -> None:
        with torch.cuda.device(self.device_id):
            with torch.cuda.stream(self.stream):
                if use_cudaMemcpy:
                    copy_cudaMemcpyAsync(self.dst, self.src, self.stream)
                else:
                    if do_reduce:
                        reduce_continous(self.dst, self.src, num_blocks, num_threads)
                    else:
                        copy_continous(self.dst, self.src, num_blocks, num_threads)


def run_perf(benches: List[Bench], use_cudaMemcpy: bool = True, do_reduce: bool = False):
    assert not (use_cudaMemcpy and do_reduce)
    for n in range(args.warmup):
        [
            bench.run(use_cudaMemcpy, do_reduce, args.num_blocks, args.num_threads)
            for bench in benches
        ]
    [bench.stream.synchronize() for bench in benches]

    [bench.start() for bench in benches]
    for n in range(args.iters):
        [
            bench.run(use_cudaMemcpy, do_reduce, args.num_blocks, args.num_threads)
            for bench in benches
        ]
    [bench.stop() for bench in benches]
    [bench.sync() for bench in benches]
    durations = [bench.duration() for bench in benches]
    for bench, duration in zip(benches, durations):
        bytes = bench.src.numel() * bench.src.dtype.itemsize * args.iters
        bw = bytes / duration / 1e6
        print(
            f"[{bench.device_id}] {bench.src.device} -> {bench.dst.device} : {duration/args.iters:0.2f} with {bw:0.2f} GB/s"
        )


if __name__ == "__main__":
    args = parse_args()
    set_cuda_p2p_access()

    copy_jobs = _parse_copy_jobs(args.copy_jobs)
    print(copy_jobs)
    dtype = DTYPE_MAP[args.dtype]

    benches = [Bench(j, (args.m, args.n), dtype) for j in copy_jobs]

    ctx = flux.get_torch_prof_ctx(args.profile)

    exp_name = args.copy_jobs.replace(":", "_").replace(",", "_")
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prof_dir = f"prof/copy_perf_{ts}"
    os.makedirs(prof_dir, exist_ok=True)

    with ctx:
        print("runing Device<->Host copy perf with cudaMemcpy")
        run_perf(benches, use_cudaMemcpy=True, do_reduce=False)
        print("runing Device<->Host copy perf with CUDA core")
        run_perf(benches, use_cudaMemcpy=False, do_reduce=False)
        if args.reduce:
            print("runing Device<->Host reduce perf with CUDA core")
            run_perf(benches, use_cudaMemcpy=False, do_reduce=True)
    if args.profile:
        ctx.export_chrome_trace(f"{prof_dir}/{exp_name}_{ts}.json.gz")
