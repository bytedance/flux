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

import torch
import torch.distributed as dist
from copy_utils import CUDA_CHECK, copy_tensor_impl

import flux
from cuda import cuda
from flux.testing import initialize_distributed, run_perf

print = partial(print, file=sys.stderr)
TP_GROUP2 = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")


def _sync_and_print(*args, **kwargs):
    torch.cuda.synchronize(torch.cuda.current_device())
    print(*args, **kwargs)


def _null_print(*args, **kwargs):
    pass


def _tensor_seg(tensor: torch.Tensor, segment: int) -> torch.Tensor:
    assert segment < WORLD_SIZE and segment >= 0
    assert tensor.shape[0] % WORLD_SIZE == 0
    m_local = tensor.shape[0] // WORLD_SIZE
    return tensor[segment * m_local : (segment + 1) * m_local, :]


def _copy_tensor(
    src_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    src_rank,
    dst_rank,
    use_cudaMemcpy,
    push,
):
    assert src_rank == RANK or dst_rank == RANK
    if dst_rank == RANK:  # make sure src_rank == RANK
        src_tensor, dst_tensor = dst_tensor, src_tensor
    if push:
        copy_tensor_impl(src_tensor, dst_tensor, use_cudaMemcpy)
    else:
        copy_tensor_impl(dst_tensor, src_tensor, use_cudaMemcpy)


def perf_2d_local_ring(
    m: int,
    n: int,
    k: int,
    copy_with_cudaMemcyAsync: bool = False,
    push: bool = True,
    wait_value: bool = True,
    write_value: bool = True,
):
    def _copy_by_ring(iter: int = -1):
        print_debug = _sync_and_print if iter == 0 else _null_print
        for numa_iter in range(NUMA_NODES):
            # copy ring intra numa
            rank_offset = (numa_node + numa_iter + 1) % NUMA_NODES * NUMA_WORLD_SIZE
            stream_inter.wait_stream(stream_intra)
            with torch.cuda.stream(stream_intra):
                for n in range(NUMA_WORLD_SIZE):
                    send_segment = (rank_local + n + 1) % NUMA_WORLD_SIZE + rank_offset
                    recv_segment = (rank_from_intra + n + 1) % NUMA_WORLD_SIZE + rank_offset
                    send_tensor = _tensor_seg(tensor_local, send_segment)
                    recv_tensor = _tensor_seg(tensor_to_intra, send_segment)
                    if n != 0:
                        if wait_value:
                            print_debug(f"wait s{send_segment} eq 1")
                            (err,) = cuda.cuStreamWaitValue32_v2(
                                stream_intra.cuda_stream,
                                _tensor_seg(flags[rank], send_segment).data_ptr(),
                                1,
                                cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
                            )
                            CUDA_CHECK(err)
                        reduce_tensor = _tensor_seg(reduce_tensors[rank], send_segment)
                        print_debug(f"add reduce s{send_segment} to r{rank} s{send_segment}")
                        send_tensor.add_(_tensor_seg(reduce_tensors[rank], send_segment))
                    if n == NUMA_WORLD_SIZE - 1:
                        break
                    print_debug(
                        f"copy intra numa s{send_segment} to r{rank_to_intra} s{send_segment} reduce"
                    )
                    _copy_tensor(
                        send_tensor,
                        recv_tensor,
                        RANK,
                        rank_to_intra,
                        copy_with_cudaMemcyAsync,
                        push,
                    )
                    if write_value:
                        print_debug(f"set r{rank_to_intra} s{send_segment} to 1")
                        (err,) = cuda.cuStreamWriteValue32_v2(
                            stream_intra.cuda_stream,
                            _tensor_seg(flags[rank_to_intra], send_segment).data_ptr(),
                            1,
                            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                        )
                        CUDA_CHECK(err)

            # copy across numa
            if numa_iter != NUMA_NODES - 1:
                stream_inter.wait_stream(stream_intra)
                with torch.cuda.stream(stream_inter):
                    print_debug(f"copy across numa #{numa_iter}")
                    send_node = (numa_node + numa_iter - 1) % NUMA_NODES
                    recv_node = (from_node + numa_iter - 1) % NUMA_NODES

                    # send prev iter done segments
                    send_segment = rank_local + send_node * NUMA_WORLD_SIZE
                    # recv from current send
                    recv_segment = rank_local + recv_node * NUMA_WORLD_SIZE
                    send_tensor = _tensor_seg(tensor_local, send_segment)
                    recv_tensor = _tensor_seg(tensor_to_inter, send_segment)
                    if wait_value:
                        print_debug(f"wait s{send_segment} eq 1")
                        (err,) = cuda.cuStreamWaitValue32_v2(
                            stream_inter.cuda_stream,
                            _tensor_seg(flags[rank], send_segment).data_ptr(),
                            1,
                            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
                        )
                        CUDA_CHECK(err)

                    print_debug(
                        f"cross numa copy {send_segment} to r{rank_to_inter} s{send_segment} reduce_inter"
                    )
                    _copy_tensor(send_tensor, recv_tensor, RANK, rank_to_inter, False, push)
                    if write_value:
                        print_debug(f"set inter r{rank_to_inter} s{send_node} to 2")
                        (err,) = cuda.cuStreamWriteValue32_v2(
                            stream_inter.cuda_stream,
                            _tensor_seg(flags_inter[rank_to_inter], send_node).data_ptr(),
                            2,
                            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                        )
                        CUDA_CHECK(err)

        # sum them all
        stream_intra.wait_stream(stream_inter)
        with torch.cuda.stream(stream_intra):
            tensor_local_out = _tensor_seg(tensor_local, rank)
            for n in range(numa_node, numa_node + NUMA_NODES - 1):
                n = n % NUMA_NODES
                segment = (n * NUMA_WORLD_SIZE + rank_local) % WORLD_SIZE
                if wait_value:
                    print_debug(f"wait reduce_inter flag {n} eq 2")
                    (err,) = cuda.cuStreamWaitValue32_v2(
                        stream_intra.cuda_stream,
                        _tensor_seg(flags_inter[rank], n).data_ptr(),
                        2,
                        cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
                    )
                    CUDA_CHECK(err)
                tensor_segment = _tensor_seg(reduce_numa_tensors[rank], segment)
                print_debug(f"add reduce_inter s{segment} to r{rank} s{segment}")
                tensor_local_out.add_(tensor_segment)
        dist.all_reduce(barrier_tensor, group=TP_GROUP, async_op=False)
        barrier_tensor.zero_()
        dist.all_reduce(barrier_tensor, group=TP_GROUP, async_op=False)

    rank = TP_GROUP.rank()
    NUMA_WORLD_SIZE = 4
    NUMA_NODES = WORLD_SIZE // NUMA_WORLD_SIZE
    rank_local = rank % NUMA_WORLD_SIZE
    numa_node = rank // NUMA_WORLD_SIZE
    next_rank_intra_numa = (rank + 1) % NUMA_WORLD_SIZE + numa_node * NUMA_WORLD_SIZE
    prev_rank_intra_numa = (
        rank - 1 + NUMA_WORLD_SIZE
    ) % NUMA_WORLD_SIZE + numa_node * NUMA_WORLD_SIZE
    next_rank_inter_numa = (rank + NUMA_WORLD_SIZE) % WORLD_SIZE
    prev_rank_inter_numa = (rank - NUMA_WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE
    rank_to_intra = prev_rank_intra_numa
    rank_from_intra = next_rank_intra_numa
    rank_to_inter = prev_rank_inter_numa
    rank_from_inter = next_rank_inter_numa
    from_node = rank_from_inter % NUMA_WORLD_SIZE
    tensor_shape = (m, n)
    tensors = flux.create_tensor_list(tensor_shape, torch.float16)
    reduce_tensors = flux.create_tensor_list(tensor_shape, torch.float16)
    reduce_numa_tensors = flux.create_tensor_list(tensor_shape, torch.float16)
    flags = flux.create_tensor_list((WORLD_SIZE, 1), torch.int32)
    flags_inter = flux.create_tensor_list((WORLD_SIZE, 1), torch.int32)
    barrier_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
    tensor_local = tensors[rank]
    tensor_to_intra = reduce_tensors[rank_to_intra]
    tensor_to_inter = reduce_numa_tensors[rank_to_inter]

    tensor_local.random_()
    tensor_local.copy_(torch.ones_like(tensor_local) * (2**RANK))
    tensor_shape = (m // WORLD_SIZE, n)
    tensor_gt = torch.zeros(tensor_shape, dtype=torch.float16, device="cuda")
    dist.reduce_scatter_tensor(tensor_gt, tensor_local, group=TP_GROUP)

    stream_intra: torch.cuda.Stream = torch.cuda.current_stream()
    stream_inter: torch.cuda.Stream = torch.cuda.Stream()
    _copy_by_ring()
    torch.cuda.synchronize(torch.cuda.current_device())
    if not torch.allclose(tensor_gt, _tensor_seg(tensor_local, rank)):
        torch.save(tensor_gt, f"gt_{RANK}.pt")
        torch.save(tensor_local, f"local_{RANK}.pt")
        torch.save(reduce_tensors[rank], f"reduce_{RANK}.pt")
        torch.save(reduce_numa_tensors[rank], f"reduce_numa_{RANK}.pt")
        raise Exception("not equal")

    run_perf(
        f"ring_2d_local",
        args.warmup,
        args.iters,
        _copy_by_ring,
        sync_per_iter=True,
    )


# 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
#


def perf_2d_opt_local_ring(
    m: int,
    n: int,
    k: int,
    copy_with_cudaMemcyAsync: bool = False,
    push: bool = True,
    wait_value: bool = True,
    write_value: bool = True,
):
    def _init_flags_async():
        dist.all_reduce(barrier_tensor, group=TP_GROUP, async_op=False)
        barrier_tensor.zero_()
        flags[rank].zero_()
        flags_inter[rank].zero_()
        dist.all_reduce(barrier_tensor, group=TP_GROUP, async_op=False)

    def _get_send_segment(iter_inter, iter_intra, ring_order):
        inter_segment = (numa_node + iter_inter + 1) % NUMA_NODES * NUMA_WORLD_SIZE
        intra_segment = (
            (rank_local + iter_intra + 1) % NUMA_WORLD_SIZE
            if ring_order == False
            else (rank_local - 1 - iter_intra + NUMA_WORLD_SIZE) % NUMA_WORLD_SIZE
        )
        return inter_segment + intra_segment

    def _copy_by_ring(iter: int = -1, ring_order=False):
        # ring_order: False for 0 <- 1 <- 2 <- 3 ... True for 0 -> 1 -> 2 -> 3 ->
        print_debug = _sync_and_print if iter == -1 else _null_print
        _init_flags_async()
        if ring_order == False:
            rank_to_intra = prev_rank_intra_numa
            rank_to_inter = prev_rank_inter_numa
        else:
            rank_to_intra = next_rank_intra_numa
            rank_to_inter = next_rank_inter_numa

        tensor_to_intra = reduce_tensors[rank_to_intra]
        tensor_to_inter = reduce_numa_tensors[rank_to_inter]
        numa_iter = 0
        for n in range(NUMA_WORLD_SIZE):
            send_segment = _get_send_segment(numa_iter, n, ring_order)
            send_tensor = _tensor_seg(tensor_local, send_segment)
            recv_tensor = _tensor_seg(tensor_to_intra, send_segment)
            if n != 0:
                if wait_value:
                    print_debug(f"[{n}] wait s{send_segment}")
                    (err,) = cuda.cuStreamWaitValue32_v2(
                        stream.cuda_stream,
                        _tensor_seg(flags[rank], send_segment).data_ptr(),
                        1,
                        cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
                    )
                    CUDA_CHECK(err)
                reduce_tensor = _tensor_seg(reduce_tensors[rank], send_segment)
                print_debug(f"[{n}] add reduce s{send_segment} to r{rank} s{send_segment}")
                send_tensor.add_(reduce_tensor)
            if n == NUMA_WORLD_SIZE - 1:
                break
            print_debug(f"[{n}] copy intra numa s{send_segment} to r{rank_to_intra} reduce")
            _copy_tensor(
                send_tensor,
                recv_tensor,
                RANK,
                rank_to_intra,
                copy_with_cudaMemcyAsync,
                push,
            )
            if write_value:
                print_debug(f"[{n}] set r{rank_to_intra} s{send_segment} to 1")
                (err,) = cuda.cuStreamWriteValue32_v2(
                    stream.cuda_stream,
                    _tensor_seg(flags[rank_to_intra], send_segment).data_ptr(),
                    1,
                    cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                )
                CUDA_CHECK(err)

        # print_debug(f"tensor_local: {tensor_local[::64, 0]}")
        # print_debug(f"tensor reduce: {reduce_tensors[rank][::64, 0]}")
        numa_iter = 1
        send_iter = 0
        for n in range(NUMA_WORLD_SIZE):
            cross_numa_send_rank = (n - 1 + NUMA_WORLD_SIZE) % NUMA_WORLD_SIZE if ring_order else n
            is_cross_numa = cross_numa_send_rank == rank_local

            if is_cross_numa:
                send_segment = (rank + NUMA_WORLD_SIZE) % WORLD_SIZE
                recv_tensor = _tensor_seg(tensor_to_inter, send_segment)
            else:
                send_segment = _get_send_segment(numa_iter, send_iter, ring_order)
                recv_tensor = _tensor_seg(tensor_to_intra, send_segment)
            send_tensor = _tensor_seg(tensor_local, send_segment)

            if (
                send_iter != 0 and not is_cross_numa
            ):  # cross numa don't have to reduce. reduce later
                if wait_value:
                    print_debug(f"[{n}] wait s{send_segment} eq 1")
                    (err,) = cuda.cuStreamWaitValue32_v2(
                        stream.cuda_stream,
                        _tensor_seg(flags[rank], send_segment).data_ptr(),
                        1,
                        cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
                    )
                    CUDA_CHECK(err)
                reduce_tensor = _tensor_seg(reduce_tensors[rank], send_segment)
                print_debug(f"[{n}] add reduce s{send_segment} to r{rank} s{send_segment}")
                send_tensor.add_(reduce_tensor)
            if is_cross_numa:
                print_debug(f"[{n}] copy inter to r{rank_to_inter} s{send_segment}")
            else:
                print_debug(f"[{n}] {rank_local} copy intra to r{rank_to_intra} s{send_segment}")

            _copy_tensor(
                send_tensor,
                recv_tensor,
                RANK,
                rank_to_intra,
                copy_with_cudaMemcyAsync,
                push,
            )
            if write_value:
                if is_cross_numa:
                    rank_to = rank_to_inter
                    flag_tensor = flags_inter[rank_to]
                    print_debug(f"[{n}] set r{rank_to} s{send_segment} to 1 cross numa")
                else:
                    rank_to = rank_to_intra
                    flag_tensor = flags[rank_to]
                    print_debug(f"[{n}] set r{rank_to} s{send_segment} to 1")
                (err,) = cuda.cuStreamWriteValue32_v2(
                    stream.cuda_stream,
                    _tensor_seg(flag_tensor, send_segment).data_ptr(),
                    1,
                    cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                )
                CUDA_CHECK(err)
            if not is_cross_numa:
                send_iter += 1

        print_debug(f"reduce local {rank}")
        if wait_value:
            (err,) = cuda.cuStreamWaitValue32_v2(
                stream.cuda_stream,
                _tensor_seg(flags[rank], rank).data_ptr(),
                1,
                cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
            )
            CUDA_CHECK(err)
            _tensor_seg(tensor_local, rank).add_(_tensor_seg(reduce_tensors[rank], rank))

        if wait_value:
            print_debug(f"wait inter flags r{rank} s{rank} eq 1")
            (err,) = cuda.cuStreamWaitValue32_v2(
                stream.cuda_stream,
                _tensor_seg(flags_inter[rank], rank).data_ptr(),
                1,
                cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
            )
            CUDA_CHECK(err)
            _tensor_seg(tensor_local, rank).add_(_tensor_seg(reduce_numa_tensors[rank], rank))
        # print_debug(f"tensor_local: {tensor_local[::64, 0]}")

    rank = TP_GROUP.rank()
    NUMA_WORLD_SIZE = 4
    NUMA_NODES = WORLD_SIZE // NUMA_WORLD_SIZE
    rank_local = rank % NUMA_WORLD_SIZE
    numa_node = rank // NUMA_WORLD_SIZE
    next_rank_intra_numa = (rank + 1) % NUMA_WORLD_SIZE + numa_node * NUMA_WORLD_SIZE
    prev_rank_intra_numa = (
        rank - 1 + NUMA_WORLD_SIZE
    ) % NUMA_WORLD_SIZE + numa_node * NUMA_WORLD_SIZE
    next_rank_inter_numa = (rank + NUMA_WORLD_SIZE) % WORLD_SIZE
    prev_rank_inter_numa = (rank - NUMA_WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE
    tensor_shape = (m, n)
    tensors = flux.create_tensor_list(tensor_shape, torch.float16)
    reduce_tensors = flux.create_tensor_list(tensor_shape, torch.float16)
    reduce_numa_tensors = flux.create_tensor_list(tensor_shape, torch.float16)
    reduce_tensors[rank].zero_()
    flags = flux.create_tensor_list((WORLD_SIZE, 1), torch.int32)
    reduce_numa_tensors[rank].zero_()
    flags_inter = flux.create_tensor_list((WORLD_SIZE, 1), torch.int32)
    barrier_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
    tensor_local = tensors[rank]

    tensor_local.random_()
    tensor_local.copy_(torch.ones_like(tensor_local) * (2**RANK))
    tensor_shape = (m // WORLD_SIZE, n)
    tensor_gt = torch.zeros(tensor_shape, dtype=torch.float16, device="cuda")
    dist.reduce_scatter_tensor(tensor_gt, tensor_local, group=TP_GROUP)

    stream: torch.cuda.Stream = torch.cuda.current_stream()
    ring_order = rank >= NUMA_WORLD_SIZE
    _copy_by_ring(-1, ring_order=ring_order)
    torch.cuda.synchronize(torch.cuda.current_device())
    if not torch.allclose(tensor_gt, _tensor_seg(tensor_local, rank)):
        torch.save(tensor_gt, f"gt_{RANK}.pt")
        torch.save(tensor_local, f"local_{RANK}.pt")
        torch.save(reduce_tensors[rank], f"reduce_{RANK}.pt")
        torch.save(reduce_numa_tensors[rank], f"reduce_numa_{RANK}.pt")
        raise Exception("not equal")

    run_perf(
        f"ring_2d_local",
        args.warmup,
        args.iters,
        partial(_copy_by_ring, ring_order=ring_order),
        sync_per_iter=True,
    )


def perf_1d_local_ring(
    m: int,
    n: int,
    k: int,
    copy_with_cudaMemcyAsync: bool = False,
    push: bool = True,
    wait_value: bool = True,
    write_value: bool = True,
):
    def _copy_by_ring(iter: int = -1):
        def _sync_and_print(*args, **kwargs):
            torch.cuda.synchronize(torch.cuda.current_device())
            print(*args, **kwargs)

        def _null_print(*args, **kwargs):
            pass

        print_debug = _sync_and_print if iter == 0 else _null_print
        # copy ring intra numa
        for n in range(LOCAL_WORLD_SIZE):
            send_segment = (rank_local + n + 1) % LOCAL_WORLD_SIZE
            recv_segment = (rank_from + n + 1) % LOCAL_WORLD_SIZE
            send_tensor = _tensor_seg(tensor_local, send_segment)
            recv_tensor = _tensor_seg(tensor_to, send_segment)
            if n != 0:
                if wait_value:
                    print_debug(f"wait s{send_segment} eq 1")
                    (err,) = cuda.cuStreamWaitValue32_v2(
                        stream.cuda_stream,
                        _tensor_seg(flags[rank], send_segment).data_ptr(),
                        1,
                        cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
                    )
                    CUDA_CHECK(err)
                reduce_tensor = _tensor_seg(reduce_tensors[rank], send_segment)
                print_debug(f"add reduce s{send_segment} to r{rank} s{send_segment}")
                send_tensor.add_(reduce_tensor)
            if n == LOCAL_WORLD_SIZE - 1:
                break
            print_debug(f"copy intra numa s{send_segment} to r{rank_to} s{send_segment} reduce")
            _copy_tensor(send_tensor, recv_tensor, RANK, rank_to, copy_with_cudaMemcyAsync, push)
            if write_value:
                print_debug(f"set r{rank_to} s{send_segment} to 1")
                (err,) = cuda.cuStreamWriteValue32_v2(
                    stream.cuda_stream,
                    _tensor_seg(flags[rank_to], send_segment).data_ptr(),
                    1,
                    cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
                )
                CUDA_CHECK(err)

        dist.all_reduce(barrier_tensor, group=TP_GROUP, async_op=False)
        barrier_tensor.zero_()
        dist.all_reduce(barrier_tensor, group=TP_GROUP, async_op=False)

    rank = TP_GROUP.rank()
    rank_local = rank % LOCAL_WORLD_SIZE
    next_rank = (rank + 1) % LOCAL_WORLD_SIZE
    prev_rank = (rank - 1 + LOCAL_WORLD_SIZE) % LOCAL_WORLD_SIZE
    rank_to = prev_rank
    rank_from = next_rank
    tensor_shape = (m, n)
    tensors = flux.create_tensor_list(tensor_shape, torch.float16)
    reduce_tensors = flux.create_tensor_list(tensor_shape, torch.float16)
    reduce_numa_tensors = flux.create_tensor_list(tensor_shape, torch.float16)
    flags = flux.create_tensor_list((WORLD_SIZE, 1), torch.int32)
    barrier_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
    tensor_local = tensors[rank]
    tensor_to = reduce_tensors[rank_to]

    tensor_local.random_()
    tensor_local.copy_(torch.ones_like(tensor_local) * (2**RANK))
    tensor_shape = (m // WORLD_SIZE, n)
    tensor_gt = torch.zeros(tensor_shape, dtype=torch.float16, device="cuda")
    dist.reduce_scatter_tensor(tensor_gt, tensor_local, group=TP_GROUP)

    stream = torch.cuda.current_stream()
    _copy_by_ring()
    torch.cuda.synchronize(torch.cuda.current_device())
    if not torch.allclose(tensor_gt, _tensor_seg(tensor_local, rank)):
        torch.save(tensor_gt, f"gt_{RANK}.pt")
        torch.save(tensor_local, f"local_{RANK}.pt")
        torch.save(reduce_tensors[rank], f"reduce_{RANK}.pt")
        torch.save(reduce_numa_tensors[rank], f"reduce_numa_{RANK}.pt")
        raise Exception("not equal")

    run_perf(
        f"ring_1d_local",
        args.warmup,
        args.iters,
        _copy_by_ring,
        sync_per_iter=True,
    )


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
        "--exp",
        choices=[
            "ring1d_single_node",
            "ring1d_multi_node",
            "ring2d_single_node",
            "ring2d_multi_node",
            "ring2d_single_node_opt",
        ],
        default="ring1d_single_node",
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
        "--wait_value",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="wait value before copy",
    )
    parser.add_argument(
        "--write_value",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="wait value before copy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    LOCAL_RANK = RANK % LOCAL_WORLD_SIZE
    args = parse_args()

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if args.exp == "ring2d_single_node":
        perf_2d_local_ring(
            args.M,
            args.N,
            args.K,
            copy_with_cudaMemcyAsync=args.with_cudaMemcpyAsync,
            push=args.push,
            wait_value=args.wait_value,
            write_value=args.write_value,
        )
    elif args.exp == "ring2d_single_node_opt":
        perf_2d_opt_local_ring(
            args.M,
            args.N,
            args.K,
            copy_with_cudaMemcyAsync=args.with_cudaMemcpyAsync,
            push=args.push,
            wait_value=args.wait_value,
            write_value=args.write_value,
        )
    elif args.exp == "ring1d_single_node":
        perf_1d_local_ring(
            args.M,
            args.N,
            args.K,
            copy_with_cudaMemcyAsync=args.with_cudaMemcpyAsync,
            push=args.push,
            wait_value=args.wait_value,
            write_value=args.write_value,
        )
