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

from typing import Union

import torch
import torch.distributed
import triton
import triton.language as tl

import flux
from flux.triton.extra import arrive_inc, tid, wait_eq


@triton.jit
def gather_rs_kernel(
    gemm_output_ptr,
    output_ptr,
    local_reduce_ptr,
    remote_reduce_ptr,
    scatter_index_ptr,
    gemm_ready_flag_ptr,
    local_barrier_ptr,
    remote_barrier_ptr,
    m_full,
    n,
    rank,
    world_size,
    num_groups: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_SPLIT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    m_per_rank = m_full // world_size
    ntokens_per_rank = m_per_rank // topk
    m_tiles_per_rank = (ntokens_per_rank + BLOCK_M - 1) // BLOCK_M
    n_per_split = n // N_SPLIT
    n_tiles_per_split = (n_per_split + BLOCK_N - 1) // BLOCK_N
    thread_idx = tid(axis=0)
    gemm_output_ptr = tl.multiple_of(gemm_output_ptr, 16)
    output_ptr = tl.multiple_of(output_ptr, 16)
    n = tl.multiple_of(n, BLOCK_N)
    for sid in range(N_SPLIT):
        wait_eq(gemm_ready_flag_ptr + sid, thread_idx, 1, scope="gpu")
        for stage in range(world_size):
            segment = (rank + stage + 1) % world_size
            for blk_id in range(pid, m_tiles_per_rank * n_tiles_per_split, num_pid):
                blk_m = blk_id // n_tiles_per_split
                blk_n = blk_id % n_tiles_per_split
                # global blk_id for barrier id
                blk_id_g = blk_id + n_tiles_per_split * m_tiles_per_rank * (
                    sid * world_size + segment
                )
                # wait flag ready
                if stage != 0:
                    wait_eq(local_barrier_ptr + blk_id_g, thread_idx, 1, scope="gpu")

                m_offset = segment * ntokens_per_rank + blk_m * BLOCK_M
                n_offset = sid * n_per_split + blk_n * BLOCK_N

                row = m_offset + tl.arange(0, BLOCK_M)
                col = n_offset + tl.arange(0, BLOCK_N)
                row_mask = row < (segment + 1) * ntokens_per_rank
                mask = row_mask[:, None] & (col[None, :] < n)
                accumulator_f32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k in range(0, topk):
                    scatter_index_ptrs = scatter_index_ptr + row * topk + k
                    scatter_index = tl.load(scatter_index_ptrs, mask=row_mask)
                    src_ptrs = gemm_output_ptr + scatter_index[:, None] * n + col[None, :]
                    for gid in range(num_groups):
                        in_ptrs = src_ptrs + gid * m_full * n
                        # in_ptrs = tl.max_contiguous(tl.multiple_of(in_ptrs, [16, 1]), [BLOCK_N, 1])
                        accumulator_f32 += tl.load(in_ptrs, mask=mask).to(tl.float32)

                accumulator = accumulator_f32.to(gemm_output_ptr.dtype.element_ty)

                row_off = (
                    tl.where(stage == world_size - 1, 0, segment * ntokens_per_rank)
                    + blk_m * BLOCK_M
                    + tl.arange(0, BLOCK_M)
                )
                output_ptrs = (
                    tl.where(stage == world_size - 1, output_ptr, remote_reduce_ptr)
                    + row_off[:, None] * n
                    + col[None, :]
                )
                # output_ptrs = tl.max_contiguous(tl.multiple_of(output_ptrs, [16, 1]), [BLOCK_N, 1])
                if stage == 0:
                    tl.store(output_ptrs, accumulator, mask=mask)
                else:
                    reduce_ptrs = local_reduce_ptr + row[:, None] * n + col[None, :]
                    # reduce_ptrs = tl.max_contiguous(
                    #     tl.multiple_of(reduce_ptrs, [16, 1]), [BLOCK_N, 1]
                    # )
                    reduce_values = tl.load(reduce_ptrs, mask=mask)
                    tl.store(output_ptrs, reduce_values + accumulator, mask=mask)

                arrive_inc(remote_barrier_ptr + blk_id_g, thread_idx, 1, scope="sys")


def gather_rs(
    gemm_output: torch.Tensor,
    output: torch.Tensor,
    local_reduce: torch.Tensor,
    remote_reduce: torch.Tensor,
    scatter_index: torch.Tensor,
    gemm_ready_flag: torch.Tensor,
    local_barrier: torch.Tensor,
    remote_barrier: torch.Tensor,
    m_full,
    n,
    num_groups,
    rank,
    world_size,
    topk: int,
    BLOCK_M: int,
    BLOCK_N: int,
    N_SPLIT: int,
):
    grid = lambda _: (4, 1, 1)
    gather_rs_kernel[grid](
        gemm_output,
        output,
        local_reduce,
        remote_reduce,
        scatter_index,
        gemm_ready_flag,
        local_barrier,
        remote_barrier,
        m_full,
        n,
        rank,
        world_size,
        num_groups,
        topk,
        BLOCK_M,
        BLOCK_N,
        N_SPLIT,
        num_warps=32,
    )


class TopkGatherRsTritonOp(torch.nn.Module):
    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        max_ntokens: int,
        hidden: int,
        topk: int,
        nexperts: int,
        output_dtype: Union[None, torch.dtype],
        BLOCK_M: int,
        BLOCK_N: int = 512,
        N_SPLIT: int = 4,
        do_all_reduce: bool = False,
    ):
        super(TopkGatherRsTritonOp, self).__init__()
        self.tp_group = tp_group
        self.rank = tp_group.rank()
        self.world_size = tp_group.size()
        self.max_ntokens = max_ntokens
        self.hidden = hidden
        self.E = nexperts
        self.topk = topk
        self.output_dtype = output_dtype
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.N_SPLIT = N_SPLIT
        self.do_all_reduce = do_all_reduce
        assert not do_all_reduce, "do_all_reduce not supported yet"
        M_tiles = (max_ntokens + self.BLOCK_M - 1) // self.BLOCK_M
        M_tiles_pad = M_tiles + self.E
        m_tiles_per_rank = (M_tiles_pad + self.world_size - 1) // self.world_size
        n_tiles_per_split = (self.hidden + self.BLOCK_N - 1) // self.BLOCK_N
        self.barrier_list = flux.create_tensor_list(
            (self.world_size * N_SPLIT * m_tiles_per_rank * n_tiles_per_split, 1),
            torch.int32,
            self.tp_group,
        )
        self.barrier = self.barrier_list[self.rank]
        self.reduce_buffer_list = flux.create_tensor_list(
            (self.max_ntokens, self.hidden), self.output_dtype, self.tp_group
        )

    def forward(
        self,
        gemm_buffer: torch.Tensor,
        outputs_buf: torch.Tensor,
        gemm_ready_flag: torch.Tensor,
        scatter_index: torch.Tensor,
        num_groups: int,
    ):
        rank_to = (self.rank + self.world_size - 1) % self.world_size
        M_full, N = gemm_buffer.shape
        assert self.hidden == N, f"self.hidden {self.hidden} != N {N}"
        assert M_full < self.max_ntokens * self.topk
        gather_rs(
            gemm_buffer,
            outputs_buf,  # TODO(houqi.1993)
            self.reduce_buffer_list[self.rank],
            self.reduce_buffer_list[rank_to],
            scatter_index,
            gemm_ready_flag,
            self.barrier_list[self.rank],
            self.barrier_list[rank_to],
            M_full,
            N,
            num_groups,
            self.rank,
            self.world_size,
            self.topk,
            BLOCK_M=self.BLOCK_M,
            BLOCK_N=512,
            N_SPLIT=self.N_SPLIT,
        )
        return outputs_buf
