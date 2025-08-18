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
import sys
from functools import partial
from typing import Callable, List, Union

import flux.testing
import torch
import torch.distributed as dist
from flux.testing import (
    init_local_groups,
    initialize_distributed,
)

import flux

print = partial(print, file=sys.stderr)


def run_perfs(exp_name: str, func: Callable, nbytes_per_iter: int):
    print(f"run exp {exp_name}")
    ctx = flux.util.get_torch_prof_ctx(True)
    with ctx:
        _, durations = flux.util.bench_func(exp_name, args.iters, args.warmups)
    ctx.export_chrome_trace(f"{exp_name}.json")
    bandwidth = nbytes_per_iter * args.iters / durations * 1e-6  # in GB/s
    print(f"{exp_name} {durations/args.iters:0.3} ms per iter, {bandwidth} GB/s")


def perf_cross_node_sendrecv_p2p(M, N, iters, warmups):
    # (node, local_rank) pair
    # (src_node, src_rank) -> (dst_node, dst_rank)
    # dst_node = (src + 1) % nnodes
    # src_node and dst_node is in range(0, local_world_size)
    src_node = RANK / LOCAL_WORLD_SIZE
    local_rank = LOCAL_RANK
    next_node = (src_node + 1) % NNODES
    prev_node = (src_node - 1 + NNODES) % NNODES

    tensor_s = torch.ones((M, N), dtype=torch.float16, device="cuda") * RANK
    tensor_r = torch.zeros_like(tensor_s)

    def to_rank(node, local_rank):
        return node * LOCAL_WORLD_SIZE + local_rank

    sender_rank, recver_rank = 0, 1
    reverse_sendrecv = False
    if reverse_sendrecv:
        sender_rank, recver_rank = 1, 0
    # p2p send/recv from (src_node, m) to (dst_node, n)
    dist.barrier()
    for send_rank in range(LOCAL_WORLD_SIZE):
        for recv_rank in range(LOCAL_WORLD_SIZE):

            def run_sendrecv():
                if local_rank == send_rank and src_node == sender_rank:
                    dist.send(tensor_s, dst=to_rank(next_node, recv_rank), group=TP_GROUP)
                elif local_rank == recv_rank and src_node == recver_rank:
                    dist.recv(tensor_r, src=to_rank(prev_node, send_rank), group=TP_GROUP)
                else:
                    # no job for node. do nothing
                    pass

            for _ in range(warmups):
                run_sendrecv()
            dist.barrier()
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iters):
                run_sendrecv()
            stop_event.record()
            stop_event.synchronize()
            elapsed = start_event.elapsed_time(stop_event)
            bandwidth = M * N * 2 * iters / elapsed * 1e-6  # in GB/s
            print(f"{elapsed/iters:0.3} ms per iter, {bandwidth} GB/s")
            dist.barrier()


def perf_sendrecv(
    M: int,
    N: int,
    use_nvshmem: bool = False,
):
    """
    test cases:
        * 0 -> 8
        * 0 -> 8 + 1 -> 9
        * 0 -> 8 + 1 -> 8
        * 0 -> 8 + 1 -> 9 + 2 -> 10
        * 0 -> 12 # cross numa node
        * all of these with bidirection
    """
    if use_nvshmem:
        tensor_send: torch.Tensor = flux.create_tensor((M, N), torch.float16)
        tensor_recv: torch.Tensor = flux.create_tensor((M, N), torch.float16)
    else:
        tensor_send = torch.ones((M, N), dtype=torch.float16, device="cuda") * RANK
        tensor_recv = torch.ones((M, N), dtype=torch.float16, device="cuda") * RANK

    def _test_case_to_next(send_rank: int, recv_rank: int, bidirectional: bool = True):
        if RANK == send_rank:
            # print(f"send to {recv_rank}")
            dist.send(tensor_send, dst=recv_rank, group=TP_GROUP)

    def _test_case_to_next_locals(
        send_ranks: List[int], recv_ranks: Union[None, List[int]] = None, bidirectional: bool = True
    ):
        if recv_ranks is None:
            recv_ranks = [(x + LOCAL_WORLD_SIZE) % WORLD_SIZE for x in send_ranks]
        for send_rank, recv_rank in zip(send_ranks, recv_ranks):
            if RANK == send_rank:
                # print(f"{RANK}: send to {recv_rank}")
                ops = [dist.P2POp(dist.isend, tensor_send, recv_rank)]
                if bidirectional:
                    ops += [dist.P2POp(dist.irecv, tensor_recv, recv_rank)]
                reqs = dist.batch_isend_irecv(ops)
                [req.wait() for req in reqs]
            elif RANK == recv_rank:
                # print(f"{RANK}: recv from {send_rank}")
                ops = [dist.P2POp(dist.irecv, tensor_recv, send_rank)]
                if bidirectional:
                    ops += [dist.P2POp(dist.isend, tensor_send, send_rank)]
                reqs = dist.batch_isend_irecv(ops)
                [req.wait() for req in reqs]

    def _test_case_all_to_all(send_ranks: List[int], recv_ranks: List[int]):
        if RANK in send_ranks:
            # print(f"send to {recv_ranks}")
            reqs = dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, tensor_send, to_rank) for to_rank in recv_ranks]
            )
            [req.wait() for req in reqs]
        elif RANK in recv_ranks:
            # print(f"recv from {send_ranks}")
            reqs = dist.batch_isend_irecv(
                [dist.P2POp(dist.irecv, tensor_recv, from_rank) for from_rank in send_ranks]
            )
            [req.wait() for req in reqs]

    nbytes_per_tensor = tensor_send.numel() * tensor_send.element_size()
    run_perfs("warmup", partial(_test_case_to_next_locals, [0]), nbytes_per_tensor)
    run_perfs("p2p_1_to_1", partial(_test_case_to_next_locals, [0]), nbytes_per_tensor)
    run_perfs(
        "p2p_1_to_1_2d",
        partial(_test_case_to_next_locals, [0], bidirectional=True),
        nbytes_per_tensor,
    )
    run_perfs("p2p_2_to_2", partial(_test_case_to_next_locals, [0, 1]), nbytes_per_tensor * 2)
    run_perfs(
        "p2p_2_to_2_2d",
        partial(_test_case_to_next_locals, [0, 1], bidirectional=True),
        nbytes_per_tensor * 2,
    )
    run_perfs("p2p_2_to_2_2numa", partial(_test_case_to_next_locals, [0, 4]), nbytes_per_tensor * 2)
    run_perfs(
        "p2p_2_to_2_2numa2d",
        partial(_test_case_to_next_locals, [0, 4], bidirectional=True),
        nbytes_per_tensor * 2,
    )
    # hangs
    run_perfs(
        "p2p_1_to_1_crossnuma", partial(_test_case_to_next_locals, [0], [15]), nbytes_per_tensor
    )
    run_perfs(
        "p2p_1_to_1_crossnuma_2d",
        partial(_test_case_to_next_locals, [0], [12], bidirectional=True),
        nbytes_per_tensor,
    )
    run_perfs("p2p_3_to_3", partial(_test_case_to_next_locals, [0, 1, 2]), nbytes_per_tensor * 3)
    run_perfs("a2a_2_to_1", partial(_test_case_all_to_all, [0, 1], [8]), nbytes_per_tensor * 2)
    run_perfs("a2a_2_to_2", partial(_test_case_all_to_all, [0, 1], [8, 9]), nbytes_per_tensor * 4)
    run_perfs("a2a_3_to_1", partial(_test_case_all_to_all, [0, 1, 2], [8]), nbytes_per_tensor * 3)
    # hang
    # run_perfs("a2a_2_to_1_numa", partial(_test_case_all_to_all, [0, 4], [8]), nbytes_per_tensor * 2)
    # run_perfs("a2a_2_to_1_numa", partial(_test_case_all_to_all, [4, 0], [8]), nbytes_per_tensor * 2)
    run_perfs(
        "a2a_2_to_2_numa", partial(_test_case_all_to_all, [0, 4], [8, 12]), nbytes_per_tensor * 4
    )


def perf_2d_ring(
    M: int,
    N: int,
    comm_inter_node: bool = True,
    comm_intra_node: bool = True,
    comm_intra_with_cudamemcpy: bool = True,
    comm_intra_push_mode: bool = False,
):
    """like what's done in ag_gemm_kernel_acrossnode:
    for cross node:
      (node, local_rank) -> (next_node, local_rank) for local_rank in range(0, local_world_size)
    for inner node:
      (node, local_rank) -> (node, next_local_rank) for local_rank in range(0, local_world_size)

    run with:
    """
    iters = args.iters
    warmups = args.warmups
    print(f"running perf_2d_ring with args: M={M}, N={N}, iters={iters}, warmups={warmups}")

    def to_rank(nodeid, local_rank):
        return nodeid * LOCAL_WORLD_SIZE + local_rank

    # p2p send/recv from (src_node, local_rank) to (dst_node, local_rank)
    def _run_sendrecv(
        allowed_ranks: List[int],
        comm_inter_node: bool,
        comm_intra_node: bool,
        comm_intra_with_cudamemcpy: bool,
    ):
        if comm_intra_node or comm_inter_node:
            dist.barrier(TP_GROUP)  # local and network not aligned. should align
        if local_rank not in allowed_ranks:
            return
        if comm_inter_node:
            print(
                f"send to {to_rank(next_node, local_rank)} and recv from {to_rank(prev_node, local_rank)}"
            )
            dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, tensor_s, to_rank(next_node, local_rank)),
                    dist.P2POp(dist.irecv, tensor_r, to_rank(prev_node, local_rank)),
                ]
            )
        if comm_intra_node:
            print(
                f"send to {to_rank(nodeid, next_local_rank)} and recv from {to_rank(nodeid, prev_local_rank)}"
            )
            if comm_intra_with_cudamemcpy:
                if comm_intra_push_mode:
                    tensor_rl_list[next_local_rank].copy_(tensor_sl)
                else:
                    tensor_sl.copy_(tensor_rl_list[next_local_rank])  # pull mode
            else:
                dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, tensor_sl, to_rank(nodeid, next_local_rank)),
                        dist.P2POp(dist.irecv, tensor_rl, to_rank(nodeid, prev_local_rank)),
                    ]
                )

    def _run_and_prof(allowed_ranks, exp_name):
        dist.barrier(TP_GROUP)
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        for n in range(warmups + iters + 1):
            _run_sendrecv(
                allowed_ranks, comm_inter_node, comm_intra_node, comm_intra_with_cudamemcpy
            )
            if n == args.warmup:
                start_event.record()
        stop_event.record()
        stop_event.synchronize()
        elapsed = start_event.elapsed_time(stop_event)
        print(elapsed)
        bandwidth = M * N * 2 * iters / elapsed * 1e-6  # in GB/s
        print(f"{elapsed/iters:0.3} ms per iter, {bandwidth} GB/s")
        dist.barrier()

    assert comm_intra_node or comm_inter_node, "should comm inter or intra node"
    nodeid = RANK // LOCAL_WORLD_SIZE
    local_rank = LOCAL_RANK
    next_node = (nodeid + 1) % NNODES
    prev_node = (nodeid - 1 + NNODES) % NNODES
    next_local_rank = (local_rank + 1) % LOCAL_WORLD_SIZE
    prev_local_rank = (local_rank - 1 + LOCAL_WORLD_SIZE) % LOCAL_WORLD_SIZE

    # s for send, r for recv, sl for send_locl, rl for recv local
    tensor_s = torch.ones((M, N), dtype=torch.float16, device="cuda") * RANK
    tensor_r = torch.zeros_like(tensor_s)
    tensor_sl = torch.ones_like(tensor_s) * RANK
    tensor_rl = torch.zeros_like(tensor_s)

    comm_intra_with_cudamemcpy = comm_intra_with_cudamemcpy and comm_intra_node

    # nccl is slow for sendrecv ring GPU-2-GPU comm. use cudaMemcpy insteadtor
    if comm_intra_node and comm_intra_with_cudamemcpy:
        init_local_groups()
        torch.cuda.synchronize()
        tensor_rl_list = [0 for _ in range(LOCAL_WORLD_SIZE)]
        # dist.barrier(TP_LOCAL_GROUP)
        print(f"group_size: {TP_LOCAL_GROUP.size()}")
        dist.all_gather_object(tensor_rl_list, tensor_rl, group=TP_LOCAL_GROUP)
    _run_and_prof(list(range(WORLD_SIZE)), "all_to_all")
    for n in range(LOCAL_WORLD_SIZE):
        allowed_ranks = list(range(n, WORLD_SIZE, LOCAL_WORLD_SIZE))
        print(f"run with rank: {allowed_ranks}")
        _run_and_prof(allowed_ranks, f"rank_{n}")


# for allagther data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("--exp", default="ring2d", choices=["ring2d", "sendrecv"])
    parser.add_argument("--warmups", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument(
        "--comm_inter_node", action="store_true", default=False, help="comm inter node"
    )
    parser.add_argument(
        "--comm_intra_node", action="store_true", default=False, help="comm intra node"
    )
    parser.add_argument(
        "--comm_intra_with_cudamemcpy",
        action="store_true",
        default=False,
        help="comm intra node with cudaMemcpy",
    )
    parser.add_argument(
        "--comm_intra_push_mode",
        action="store_true",
        default=False,
        help="comm intra node with cudaMemcpy",
    )
    parser.add_argument(
        "--use_nvshmem",
        action="store_true",
        default=False,
        help="comm intra node with cudaMemcpy",
    )
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument("--push", default=False, action="store_true", help="push mode")
    return parser.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    TP_LOCAL_GROUP = flux.testing.init_local_groups()
    assert TP_LOCAL_GROUP is not None
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    LOCAL_RANK = RANK % LOCAL_WORLD_SIZE
    args = parse_args()
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    if args.exp == "ring2d":
        perf_2d_ring(
            args.M,
            args.N,
            comm_inter_node=args.comm_inter_node,
            comm_intra_node=args.comm_intra_node,
            comm_intra_push_mode=args.comm_intra_push_mode,
        )
    elif args.exp == "sendrecv":
        perf_sendrecv(args.M, args.N, args.use_nvshmem)
    else:
        raise Exception(f"unsupported exp {args.exp}")
