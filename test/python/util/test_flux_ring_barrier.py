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

# usage: torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 --rdzv_id=none --master_addr=127.0.0.1 --master_port=23456 test/test_flux_ring_barrier.py
import argparse
import time

import torch
import torch.distributed

import flux
from flux.testing import initialize_distributed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ring_mode", default=False, action="store_true")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    LOCAL_RANK = RANK % LOCAL_WORLD_SIZE

    ctx = flux.util.get_torch_prof_ctx(args.profile)

    print(f"GroupBarrier: {args.ring_mode}", flush=True)
    group_barrier = flux.GroupBarrier(TP_GROUP, args.ring_mode)

    TP_GROUP.barrier()
    t1 = time.time()
    torch.cuda.synchronize()
    stream = torch.cuda.current_stream()
    with ctx:
        for i in range(args.iters):
            group_barrier.barrier_all(stream.cuda_stream)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Done in {t2 - t1}s", flush=True)
    if args.profile:
        ctx.export_chrome_trace(f"prof/trace_rank{TP_GROUP.rank()}.json.gz")
