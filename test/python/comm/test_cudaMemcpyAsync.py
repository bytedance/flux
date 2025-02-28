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

# Usage: torchrun --nnodes=1 --nproc_per_node=8 test/python/comm/test_cudaMemcpyAsync.py 2048 1024
import argparse
import sys
from functools import partial

import torch
import torch.distributed

import flux
from flux.testing import initialize_distributed, run_perf

print = partial(print, file=sys.stderr)


def copy_all_to_all():
    def _tensor_seg(tensor, seg_id):
        assert seg_id < WORLD_SIZE and seg_id >= 0
        assert tensor.shape == tensor_shape
        m, _ = tensor.shape
        m_per_seg = m // WORLD_SIZE
        return tensor[m_per_seg * seg_id : m_per_seg * (seg_id + 1), :]

    def _copy_all_to_all(iter: int):
        [s.wait_stream(current_stream) for rank, s in enumerate(streams) if rank != RANK]
        for to_rank in range(RANK + 1, RANK + WORLD_SIZE):
            to_rank = to_rank % WORLD_SIZE
            # copy to all other ranks
            torch.cuda.set_stream(streams[to_rank])
            local_tensor = _tensor_seg(tensors[RANK], to_rank)
            remote_tensor = _tensor_seg(tensors[to_rank], RANK)
            if args.push:
                remote_tensor.copy_(local_tensor)
            else:
                local_tensor.copy_(remote_tensor)
        [current_stream.wait_stream(s) for rank, s in enumerate(streams) if rank != RANK]
        torch.cuda.set_stream(current_stream)

    streams = [torch.cuda.Stream() for _ in range(WORLD_SIZE)]
    current_stream = torch.cuda.current_stream()

    tensor_shape = (args.M, args.K)
    tensors = flux.create_tensor_list(tensor_shape, torch.float16, TP_GROUP)
    run_perf("copy_all_to_all", args.warmup, args.iters, _copy_all_to_all, sync_per_iter=True)


def _copy_ring(use_nvshmem: bool = False, strided: bool = False):
    def _get_tensor():
        if use_nvshmem:
            tensors = flux.create_tensor_list(tensor_shape, torch.float16, TP_GROUP)
            src_tensor = tensors[src_rank]
            dst_tensor = tensors[dst_rank]
            return src_tensor, dst_tensor
        else:
            src_tensor = torch.zeros(tensor_shape, dtype=torch.float16, device=f"cuda:{src_rank}")
            dst_tensor = torch.zeros(tensor_shape, dtype=torch.float16, device=f"cuda:{dst_rank}")
        return src_tensor, dst_tensor

    def _copy(iter: int = 0):
        if args.push:
            dst_tensor.copy_(src_tensor)
        else:
            src_tensor.copy_(dst_tensor)

    def exp_name():
        return "copy_ring_{}+{}".format(
            "local" if not use_nvshmem else "nvshmem", "continous" if not strided else "strided"
        )

    src_rank = TP_GROUP.rank()
    dst_rank = (src_rank + 1) % WORLD_SIZE  # send to next_rank
    if strided:
        tensor_shape = (args.M // WORLD_SIZE, args.K * 2)
        src_tensor, dst_tensor = _get_tensor()
        src_tensor = src_tensor[:, : args.K]
        dst_tensor = dst_tensor[:, : args.K]
    else:
        tensor_shape = (args.M // WORLD_SIZE, args.K)
        src_tensor, dst_tensor = _get_tensor()
    run_perf(
        exp_name(),
        args.warmup,
        args.iters,
        _copy,
        sync_per_iter=True,
    )


def copy_ring():
    print("local+continous")
    _copy_ring(use_nvshmem=False, strided=False)

    print("nvshmem+continous")
    _copy_ring(use_nvshmem=True, strided=False)

    print("local+strided")
    _copy_ring(use_nvshmem=False, strided=True)

    print("nvshmem+strided")
    _copy_ring(use_nvshmem=True, strided=True)


# for allagther data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument(
        "--exp", default="copy_ring", choices=["copy_ring", "copy_all_to_all", "all"]
    )
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--strided", default=False, action="store_true")
    parser.add_argument("--push", default=False, action="store_true", help="push mode")
    return parser.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()
    args = parse_args()

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if args.exp in ["copy_ring", "all"]:
        copy_ring()
    if args.exp in ["copy_all_to_all", "all"]:
        copy_all_to_all()
