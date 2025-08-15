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
# Usage: ./launch.sh test/python/ag_gemm/test_comm_ag.py 1024 1024 1024

import argparse
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
from flux.testing import (
    NNODES,
    initialize_distributed,
    run_perf,
)

import flux

print = partial(print, file=sys.stderr)


def perf_1d_ring(
    M: int,
    N: int,
    K: int,
    push_mode: bool = False,
):
    """
    run with topo:  0 <- 1 <- 2 <- 3 <- ... <- 15 <- 0
    """

    def _tensor_seg(tensor, segment):
        assert tensor.shape[0] == M
        assert segment < WORLD_SIZE
        m_per_seg = M // WORLD_SIZE
        return tensor[m_per_seg * segment : m_per_seg * (segment + 1), ...]

    def _init_tensor(tensor):
        tensor.random_()
        return tensor

    flag_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")  # only for send/recv sync

    def _run_sendrecv(iter: int = 0):
        print_debug = print if iter == 0 else lambda *args, **kwargs: None
        for iter in range(WORLD_SIZE - 1):
            send_segment = (iter + RANK) % WORLD_SIZE
            recv_segment = (iter + RANK + 1) % WORLD_SIZE
            send_tensor = _tensor_seg(tensor_local, send_segment)
            recv_tensor = _tensor_seg(tensor_local, recv_segment)
            send_reqs, recv_reqs = [], []
            if inter_node_recv:
                print_debug(f"iter {iter:02} recv from {from_rank}")
                recv_reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.irecv, recv_tensor, from_rank, TP_GROUP),
                    ]
                )
            if inter_node_send:
                print_debug(f"iter {iter:02} send to {to_rank}")
                send_reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, send_tensor, to_rank, TP_GROUP),
                    ]
                )
            else:  # copy with cudaMemcpy
                if push_mode:
                    _tensor_seg(tensor_to, send_segment).copy_(send_tensor)
                else:
                    recv_tensor.copy_(_tensor_seg(tensor_from, recv_segment))
            [req.wait() for req in send_reqs + recv_reqs]
            dist.all_reduce(flag_tensor, group=TP_GROUP, async_op=False)

    iters = args.iters
    warmup_iters = args.warmup
    print(f"run perf_1d_ring with args: M={M}, N={N}, K={K}, iters={iters}, warmup={warmup_iters}")

    # p2p send/recv from (src_node, local_rank) to (dst_node, local_rank)
    inter_node_send: bool = RANK % LOCAL_WORLD_SIZE == 0
    inter_node_recv: bool = (RANK + 1) % LOCAL_WORLD_SIZE == 0
    local_rank = LOCAL_RANK
    next_rank = (RANK + 1) % WORLD_SIZE
    prev_rank = (RANK - 1 + WORLD_SIZE) % WORLD_SIZE
    from_rank = next_rank
    to_rank = prev_rank

    # s for send, r for recv, sl for send_locl, rl for recv local
    tensors = flux.create_tensor_list((M, K), torch.float16, TP_GROUP)
    tensor_local = tensors[local_rank % LOCAL_WORLD_SIZE]
    tensor_to = tensors[to_rank % LOCAL_WORLD_SIZE]  # NOTE: don't use this for inter node comm
    tensor_from = tensors[from_rank % LOCAL_WORLD_SIZE]  # NOTE: don't use this for inter node comm

    _init_tensor(_tensor_seg(tensor_local, RANK))
    tensor_gt = torch.zeros_like(tensor_local)
    dist.all_gather_into_tensor(tensor_gt, _tensor_seg(tensor_local, RANK), group=TP_GROUP)
    # nccl is slow for sendrecv ring GPU-2-GPU comm. use cudaMemcpy insteadtor
    dist.barrier(TP_GROUP)
    time.sleep(1)
    torch.cuda.synchronize()
    _run_sendrecv()
    # dist.barrier(TP_GROUP)
    # time.sleep(1)
    assert torch.allclose(tensor_gt, tensor_local)
    run_perf("ring_1d", args.warmup, args.iters, _run_sendrecv, sync_per_iter=True)


def perf_1dx2_ring(
    M: int,
    N: int,
    K: int,
    push_mode: bool = False,
    with_cudaMemcpyAsync: bool = False,
):
    """
    run with topo: 0 <- 1 <- 2 <- 3 <- 8 <- 9 <- 10 <- 11 <- ...
    run with topo: 0 <- 1 <- 2 <- 3 <- 12 <- 13 <- 14 <- 15 <- ...
    """

    def _tensor_seg(tensor, segment, factor=2):
        assert tensor.shape[0] == M
        assert segment < (WORLD_SIZE * factor)
        m_per_seg = M // (WORLD_SIZE * factor)
        return tensor[m_per_seg * segment : m_per_seg * (segment + 1), ...]

    def _init_tensor(tensor):
        tensor.random_()
        return tensor

    flag_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")  # only for send/recv sync

    RING_OFFSET = 2

    def _rank_trans(rank, rank_offset):
        local_rank, node_idx = rank % LOCAL_WORLD_SIZE, rank // LOCAL_WORLD_SIZE
        local_rank_trans = (local_rank + rank_offset) % LOCAL_WORLD_SIZE
        return local_rank_trans + node_idx * LOCAL_WORLD_SIZE

    def _rank_inverse(rank, rank_offset):
        local_rank, node_idx = rank % LOCAL_WORLD_SIZE, rank // LOCAL_WORLD_SIZE
        local_rank_trans = (local_rank - rank_offset + LOCAL_WORLD_SIZE) % LOCAL_WORLD_SIZE
        return local_rank_trans + node_idx * LOCAL_WORLD_SIZE

    def _caculate_segments(rank_offset):
        send_segments = [[-1 for _ in range(WORLD_SIZE)] for _ in range(WORLD_SIZE)]
        recv_segments = [[-1 for _ in range(WORLD_SIZE)] for _ in range(WORLD_SIZE)]
        for iter in range(WORLD_SIZE - 1):
            for rank in range(WORLD_SIZE):
                rank_from = _rank_inverse(
                    (_rank_trans(rank, rank_offset) + 1) % WORLD_SIZE, rank_offset
                )
                if iter == 0:
                    send_segment = (rank + iter) % WORLD_SIZE
                else:
                    send_segment = recv_segments[rank][iter - 1]
                send_segments[rank][iter] = send_segment
            for rank in range(WORLD_SIZE):
                rank_from = _rank_inverse(
                    (_rank_trans(rank, rank_offset) + 1) % WORLD_SIZE, rank_offset
                )
                recv_segment = send_segments[rank_from][iter]
                recv_segments[rank][iter] = recv_segment
        return send_segments, recv_segments

    class sendrecv_1d_helper(object):
        # normal ring

        def __init__(self, tensors):
            self.rank = RANK
            self.inter_node_send: bool = self.rank % LOCAL_WORLD_SIZE == 0
            self.inter_node_recv: bool = (self.rank + 1) % LOCAL_WORLD_SIZE == 0
            next_rank = (self.rank + 1) % WORLD_SIZE
            prev_rank = (self.rank - 1 + WORLD_SIZE) % WORLD_SIZE
            self.from_rank = next_rank
            self.to_rank = prev_rank
            self.tensor_to = tensors[
                self.to_rank % LOCAL_WORLD_SIZE
            ]  # NOTE: don't use this for inter node comm
            self.tensor_from = tensors[
                self.from_rank % LOCAL_WORLD_SIZE
            ]  # NOTE: don't use this for inter node comm

        def run_iter(self, iter, iter_i, pg=TP_GROUP):
            print_debug = print if iter == 0 and iter_i == 0 else lambda *args, **kwargs: None
            send_segment = (iter_i + self.rank) % WORLD_SIZE
            recv_segment = (iter_i + self.rank + 1) % WORLD_SIZE
            send_segment = send_segment * 2
            recv_segment = recv_segment * 2
            send_tensor = _tensor_seg(tensor_local, send_segment)
            recv_tensor = _tensor_seg(tensor_local, recv_segment)
            ops = []
            if with_cudaMemcpyAsync:
                if self.inter_node_recv:
                    print_debug(f"iter {iter_i:02} recv from {self.from_rank}")
                    ops.append(dist.P2POp(dist.irecv, recv_tensor, self.from_rank, pg))
                    [req.wait() for req in dist.batch_isend_irecv(ops)]
                if self.inter_node_send:
                    print_debug(f"iter {iter_i:02} send to {self.to_rank}")
                    ops.append(dist.P2POp(dist.isend, send_tensor, self.to_rank, pg))
                    [req.wait() for req in dist.batch_isend_irecv(ops)]
                else:  # copy with cudaMemcpy
                    if push_mode:
                        _tensor_seg(self.tensor_to, send_segment).copy_(send_tensor)
                    else:
                        recv_tensor.copy_(_tensor_seg(self.tensor_from, recv_segment))
            else:
                # add a send and a recv
                ops.append(dist.P2POp(dist.irecv, recv_tensor, self.from_rank, pg))
                ops.append(dist.P2POp(dist.isend, send_tensor, self.to_rank, pg))
            return ops

    class sendrecv_1d_numa_helper(object):
        def __init__(self, tensors, rank_offset):
            rank_trans = _rank_trans(RANK, rank_offset)
            self.inter_node_send: bool = rank_trans % LOCAL_WORLD_SIZE == 0
            self.inter_node_recv: bool = (rank_trans + 1) % LOCAL_WORLD_SIZE == 0
            next_rank_trans = (rank_trans + 1) % WORLD_SIZE
            prev_rank_trans = (rank_trans - 1 + WORLD_SIZE) % WORLD_SIZE
            self.from_rank = _rank_inverse(next_rank_trans, rank_offset)
            self.to_rank = _rank_inverse(prev_rank_trans, rank_offset)
            self.tensor_to = tensors[
                self.to_rank % LOCAL_WORLD_SIZE
            ]  # NOTE: don't use this for inter node comm
            self.tensor_from = tensors[
                self.from_rank % LOCAL_WORLD_SIZE
            ]  # NOTE: don't use this for inter node comm

        def run_iter(self, iter, iter_i, pg=TP_GROUP):
            print_debug = print if iter == 0 and iter_i == 0 else lambda *args, **kwargs: None
            send_segment = send_segments[RANK][iter_i]
            recv_segment = recv_segments[RANK][iter_i]
            send_segment = send_segment * 2 + 1
            recv_segment = recv_segment * 2 + 1
            send_tensor = _tensor_seg(tensor_local, send_segment)
            recv_tensor = _tensor_seg(tensor_local, recv_segment)
            ops = []
            if iter_i == 0:
                print(f"iter {iter_i:02} send to {self.to_rank} recv from {self.from_rank}")
            if with_cudaMemcpyAsync:
                if self.inter_node_recv:
                    print_debug(f"iter {iter_i:02} recv from {self.from_rank}")
                    ops.append(dist.P2POp(dist.irecv, recv_tensor, self.from_rank, pg))
                    [req.wait() for req in dist.batch_isend_irecv(ops)]
                if self.inter_node_send:
                    print_debug(f"iter {iter_i:02} send to {self.to_rank}")
                    ops.append(dist.P2POp(dist.isend, send_tensor, self.to_rank, pg))
                    [req.wait() for req in dist.batch_isend_irecv(ops)]
                else:  # copy with cudaMemcpy
                    if push_mode:
                        _tensor_seg(self.tensor_to, send_segment).copy_(send_tensor)
                    else:
                        recv_tensor.copy_(_tensor_seg(self.tensor_from, recv_segment))
            else:
                # add a send and a recv
                ops.append(dist.P2POp(dist.irecv, recv_tensor, self.from_rank, pg))
                ops.append(dist.P2POp(dist.isend, send_tensor, self.to_rank, pg))
            return ops

    def _run_sendrecv_serialize(
        ring1: sendrecv_1d_helper, ring2: sendrecv_1d_numa_helper, iter: int = 0
    ):
        for iter_i in range(WORLD_SIZE - 1):
            reqs1 = dist.batch_isend_irecv(ring1.run_iter(iter, iter_i, TP_GROUP))
            reqs2 = dist.batch_isend_irecv(ring2.run_iter(iter, iter_i, TP_GROUP2))
            # print(ops)
            [req.wait() for req in reqs1]
            [req.wait() for req in reqs2]
            if with_cudaMemcpyAsync:
                dist.all_reduce(flag_tensor, group=TP_GROUP, async_op=False)

    def _run_sendrecv_parallel(ring1: sendrecv_1d_helper, ring2: sendrecv_1d_numa_helper):
        for iter in range(WORLD_SIZE - 1):
            ring1_stream.wait_stream(current_stream)
            ring2_stream.wait_stream(current_stream)
            with torch.cuda.stream(ring1_stream):
                ring1.run_iter(iter)
            with torch.cuda.stream(ring2_stream):
                ring2.run_iter(iter)
            current_stream.wait_stream(ring1_stream)
            current_stream.wait_stream(ring2_stream)
            dist.all_reduce(flag_tensor, group=TP_GROUP, async_op=False)

    def _run_sendrecv(ring1, ring2, iter: int = 0):
        if not with_cudaMemcpyAsync:
            _run_sendrecv_serialize(ring1, ring2)
        else:
            _run_sendrecv_parallel(ring1, ring2)

    iters = args.iters
    warmup_iters = args.warmup
    print(f"perf_1dx2_ring with args: M={M}, N={N}, K={K}, iters={iters}, warmup={warmup_iters}")

    # s for send, r for recv, sl for send_locl, rl for recv local
    tensors = flux.create_tensor_list((M, K), torch.float16, TP_GROUP)
    tensor_local = tensors[LOCAL_RANK]

    ring1_stream, ring2_stream = torch.cuda.Stream(), torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()

    for rank_offset in range(1, LOCAL_WORLD_SIZE - 1):
        send_segments, recv_segments = _caculate_segments(rank_offset)
        ring1 = sendrecv_1d_helper(tensors=tensors)
        ring2 = sendrecv_1d_numa_helper(tensors=tensors, rank_offset=rank_offset)

        _init_tensor(_tensor_seg(tensor_local, RANK, factor=1))
        tensor_gt = torch.zeros_like(tensor_local)
        dist.all_gather_into_tensor(
            tensor_gt, _tensor_seg(tensor_local, RANK, factor=1), group=TP_GROUP
        )
        # nccl is slow for sendrecv ring GPU-2-GPU comm. use cudaMemcpy insteadtor
        dist.barrier(TP_GROUP)
        time.sleep(1)
        torch.cuda.synchronize()
        _run_sendrecv(ring1, ring2)
        assert torch.allclose(tensor_gt, tensor_local)
        run_perf(
            f"ring_1d_{rank_offset}",
            args.warmup,
            args.iters,
            partial(_run_sendrecv, ring1, ring2),
            sync_per_iter=True,
        )


def perf_2d_ring(
    M: int,
    N: int,
    K: int,
    with_cudaMemcpyAsync: bool,
    push_mode: bool = False,
    no_cudaMemcpyAsync_cross_numa: bool = False,
    opt: bool = False,
):
    def _tensor_seg(tensor, rank):
        assert rank < WORLD_SIZE and rank >= 0
        assert tensor.shape[0] % WORLD_SIZE == 0
        m_local = tensor.shape[0] // WORLD_SIZE
        return tensor[rank * m_local : (rank + 1) * m_local, :]

    def rank_at(rank, node):
        local_rank = (rank + LOCAL_WORLD_SIZE) % LOCAL_WORLD_SIZE
        return local_rank + node * LOCAL_WORLD_SIZE

    def _init_tensor(tensor):
        tensor.copy_(torch.ones_like(tensor) * RANK)

    def _run():
        for iter in range(NNODES):
            # with send_recv with nccl
            inter_reqs = []
            if iter != NNODES - 1:
                send_node = (NODE_ID + iter) % NNODES
                recv_node = (node_from + iter) % NNODES
                send_segment = rank_at(RANK, send_node)
                send_tensor = _tensor_seg(tensor_local, send_segment)
                recv_segment = rank_at(RANK, recv_node)
                recv_tensor = _tensor_seg(tensor_local, recv_segment)
                print(
                    f"iter {iter:2} inter node send {send_segment} to {rank_to_node} recv {recv_segment} from {rank_from_node}"
                )
                # run communicate
                inter_reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, send_tensor, rank_to_node, TP_GROUP2),
                        dist.P2POp(dist.irecv, recv_tensor, rank_from_node, TP_GROUP2),
                    ]
                )

            intra_reqs = []
            for n in range(LOCAL_WORLD_SIZE - 1):
                intra_comm_node = (NODE_ID + iter) % NNODES
                # local all_gather
                send_segment = rank_at(RANK + n, intra_comm_node)
                recv_segment = rank_at(rank_from_local + n, intra_comm_node)
                send_tensor = _tensor_seg(tensor_local, send_segment)
                recv_tensor = _tensor_seg(tensor_local, recv_segment)
                print(
                    f"iter {iter:2}/{n:2} intra node send {send_segment} to {rank_to_local} recv {recv_segment} from {rank_from_local} push_mode: {push_mode}"
                )
                ops = []
                with_cudaMemcpyAsync_this_iter = with_cudaMemcpyAsync or (
                    opt and iter == NNODES - 1
                )
                if with_cudaMemcpyAsync_this_iter:
                    if push_mode:
                        remote_send_tensor = _tensor_seg(
                            tensors[rank_to_local % LOCAL_WORLD_SIZE], send_segment
                        )
                        remote_send_tensor.copy_(send_tensor)
                    else:
                        remote_recv_tensor = _tensor_seg(
                            tensors[rank_from_local % LOCAL_WORLD_SIZE], recv_segment
                        )
                        recv_tensor.copy_(remote_recv_tensor)
                    work = dist.all_reduce(flag_tensor, group=TP_GROUP, async_op=True)
                    # intra_reqs.append(work)
                    work.wait()
                else:
                    ops.append(dist.P2POp(dist.irecv, recv_tensor, rank_from_local, TP_GROUP))
                    ops.append(dist.P2POp(dist.isend, send_tensor, rank_to_local, TP_GROUP))
                    reqs = dist.batch_isend_irecv(ops)
                    intra_reqs.extend(reqs)
            [req.wait() for req in intra_reqs]
            [req.wait() for req in inter_reqs]

    tensors = flux.create_tensor_list((M, K), torch.float16, TP_GROUP)
    tensor_local = tensors[LOCAL_RANK]
    _init_tensor(_tensor_seg(tensor_local, RANK))
    tensor_gt = torch.zeros_like(tensor_local)
    dist.all_gather_into_tensor(tensor_gt, _tensor_seg(tensor_local, RANK), group=TP_GROUP)
    # nccl is slow for sendrecv ring GPU-2-GPU comm. use cudaMemcpy insteadtor
    dist.barrier(TP_GROUP)
    time.sleep(1)
    torch.cuda.synchronize()

    rank_prev_local = (RANK - 1 + LOCAL_WORLD_SIZE) % LOCAL_WORLD_SIZE + NODE_ID * LOCAL_WORLD_SIZE
    rank_next_local = (RANK + 1) % LOCAL_WORLD_SIZE + NODE_ID * LOCAL_WORLD_SIZE
    rank_to_local = rank_prev_local
    rank_from_local = rank_next_local
    rank_prev_node = (RANK - LOCAL_WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE
    rank_next_node = (RANK + LOCAL_WORLD_SIZE) % WORLD_SIZE
    rank_from_node = rank_prev_node
    rank_to_node = rank_next_node
    node_prev = (NODE_ID - 1 + NNODES) % NNODES
    node_next = (NODE_ID + 1) % NNODES
    node_from = node_next
    node_to = node_prev
    flag_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")  # only for send/recv sync
    _run()
    assert torch.allclose(tensor_gt, tensor_local)
    run_perf(
        f"ring_2d",
        args.warmup,
        args.iters,
        _run,
        sync_per_iter=True,
    )


def perf_2d_ring_local(tensor_full):
    pass


# for allagther data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument(
        "--profile",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="dump torch.profiler.profile",
    )
    parser.add_argument(
        "--push", default=False, action=argparse.BooleanOptionalAction, help="push mode"
    )
    parser.add_argument(
        "--with_cudaMemcpyAsync",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="copy intra numa with cudaMemcpyAsync",
    )
    parser.add_argument(
        "--exp",
        choices=["ring1d", "ring1dx2", "ring2d"],
        default="ring1d",
        help="ring1d is a 1d ring, ring1dx2 is 2 1d ring",
    )
    parser.add_argument(
        "--no_cudaMemcpyAsync_cross_numa",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="copy intra numa with cudaMemcpyAsync. only valid with --with_cudaMemcpyAsync",
    )
    parser.add_argument(
        "--opt",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use optimized ring",
    )
    return parser.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    LOCAL_RANK = RANK % LOCAL_WORLD_SIZE
    NODE_ID = RANK // LOCAL_WORLD_SIZE
    TP_GROUP2 = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    args = parse_args()

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if args.exp == "ring1d":
        perf_1d_ring(
            args.M,
            args.N,
            args.K,
            push_mode=args.push,
        )
    elif args.exp == "ring1dx2":
        perf_1dx2_ring(
            args.M,
            args.N,
            args.K,
            push_mode=args.push,
            with_cudaMemcpyAsync=args.with_cudaMemcpyAsync,
        )
    elif args.exp == "ring2d":
        perf_2d_ring(
            args.M,
            args.N,
            args.K,
            with_cudaMemcpyAsync=args.with_cudaMemcpyAsync,
            push_mode=args.push,
            no_cudaMemcpyAsync_cross_numa=args.no_cudaMemcpyAsync_cross_numa,
            opt=args.opt,
        )
