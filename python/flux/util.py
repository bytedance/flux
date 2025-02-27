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

import gzip
import json
import logging
import pathlib
import shutil
import sys
from contextlib import contextmanager, nullcontext
from typing import List, Optional

import torch
import torch.distributed


def get_arch():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    major = properties.major
    minor = properties.minor
    return major * 10 + minor


def torch_allclose(x, y, rtol, atol, verbose=True):
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

        if len(y.shape) == 1:
            diff_rate = num_diff / y.shape[0]
        else:
            diff_rate = num_diff / (y.shape[0] * y.shape[1])
        print(f"diff count: {num_diff} ({diff_rate*100:.3f}%), {list(y.shape)}", file=sys.stderr)
        max_diff = torch.max(torch.abs(x - y))
        rtol_abs = rtol * torch.min(torch.abs(y))
        print(f"diff max: {max_diff}, atol: {atol}, rtol_abs: {rtol_abs}", file=sys.stderr)
        diff_indices = (diff_loc == True).nonzero(as_tuple=False)
        print(f"diff locations:\n{diff_indices}", file=sys.stderr)
        print("--------------------------------------------------------------\n", file=sys.stderr)
        raise RuntimeError

    if verbose:
        print("all close!")


def get_torch_prof_ctx(do_prof: bool):
    ctx = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        )
        if do_prof
        else nullcontext()
    )
    return ctx


def _merge_json(to_merge_files: List[pathlib.Path], output_json: pathlib.Path):
    events = []
    for json_file in to_merge_files:
        if str(json_file).endswith("merged.json"):
            continue
        with open(json_file, "rb") as f:
            logging.debug(f"merge {json_file}")
            full_tl_json = json.loads(f.read().decode("latin-1"), strict=False)

        rank = full_tl_json["distributedInfo"]["rank"]
        world_size = full_tl_json["distributedInfo"]["world_size"]
        for e in full_tl_json["traceEvents"]:
            e["pid"] = f"{e['pid']}_{rank}"
            if isinstance(e["tid"], int):
                e["tid"] = e["tid"] * world_size + rank
            if e["name"] == "thread_name":
                e["args"]["name"] = f'{e["args"]["name"]}_{rank}'
            if e["name"] == "thread_sort_index":  # perfetto does not respect this.
                e["args"]["sort_index"] = e["args"]["sort_index"] * world_size + rank
        events.extend(full_tl_json["traceEvents"])

    with open(output_json, "w") as f:
        full_tl_json["traceEvents"] = events
        json.dump(events, f)


class group_profile:
    def __init__(
        self,
        name: str = None,
        do_prof: bool = True,
        merge_group: bool = True,
        keep_merged_only: bool = True,
        compress: bool = True,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.name = name
        self.do_prof = do_prof
        self.profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        )
        self.group = group or torch.distributed.group.WORLD
        self.merge_group = merge_group
        self.keep_merged_only = keep_merged_only
        self.compress = compress

    def __enter__(self):
        if self.do_prof:
            self.profile.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.do_prof:
            self.profile.__exit__(exc_type, exc_val, exc_tb)
            # export chrome trace
            outfile = pathlib.Path("prof") / f"{self.name}" / f"rank{self.group.rank()}.json"
            outfile.mkdir(parents=True, exist_ok=True)
            print(f"export chrome trace to {outfile}")
            self.profile.export_chrome_trace(str(outfile))
            if self.merge_group:
                self.merge_all()

    def merge_all(self):
        print(f"merge profiles...")
        # merge all
        if self.merge_group:
            torch.cuda.synchronize()  # wait for all ranks export
            torch.distributed.barrier(group=self.group)
            torch.cuda.synchronize()  # wait for all ranks export
        if self.group.rank() != 0:
            return

        # merge all json
        outdir = pathlib.Path("prof") / f"{self.name}"
        to_merge_files = outdir.glob("*.json")
        merged_json = pathlib.Path("prof") / f"{self.name}_merged.json"
        _merge_json(to_merge_files, merged_json)
        print(f"merge all profiles into {merged_json}")
        # here is an issue with python 3.10+ & with_stack=True save gz failure: https://github.com/pytorch/pytorch/issues/113564
        if self.compress:
            with open(merged_json, "rb") as f:
                with gzip.open(merged_json.with_suffix(".json.gz"), "wb") as g:
                    g.write(f.read())
            merged_json.unlink()
        if self.keep_merged_only:
            logging.info(f"remove profile directory: {outdir}")
            shutil.rmtree(outdir)


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1 and dtype.is_floating_point


@contextmanager
def with_torch_deterministic(mode: bool, warn_only: bool = True):
    old_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_mode, warn_only=warn_only)


def bench_func(func, iters, warmup_iters):
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


__all__ = [
    "is_fp8_dtype",
    "get_arch",
    "torch_allclose",
    "get_torch_prof_ctx",
    "with_torch_deterministic",
    "bench_func",
    "group_profile",
]
