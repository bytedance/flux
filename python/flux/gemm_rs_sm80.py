################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

from functools import partial
import torch
import torch.distributed as dist
import flux
from typing import Optional

FLUX_GEMM_RS_INTER_NODE_GROUP = None
INTRA_NODE_TP_GROUP = None

print = partial(print, flush=True)


def get_intra_node_pg_group(tp_group: torch.distributed.ProcessGroup, nnodes: int):
    global INTRA_NODE_TP_GROUP
    if INTRA_NODE_TP_GROUP is not None:
        return INTRA_NODE_TP_GROUP

    tp_world_size: int = tp_group.size()
    assert tp_world_size % nnodes == 0, f"{tp_world_size} is not divisible by {nnodes}"
    local_world_size: int = tp_group.size() // nnodes
    world_size: int = torch.distributed.get_world_size()
    assert world_size % tp_world_size == 0, f"{world_size} not divisible by {tp_world_size}"

    ranks_per_sub_group = []
    for node_id in range(nnodes):
        ranks_per_sub_group.append(
            list(range(node_id * local_world_size, (node_id + 1) * local_world_size))
        )

    INTRA_NODE_TP_GROUP, _ = torch.distributed.new_subgroups_by_enumeration(ranks_per_sub_group)
    print(f"split tp_group({tp_group.size()}) into {nnodes}x{INTRA_NODE_TP_GROUP.size()} subgroups")
    return INTRA_NODE_TP_GROUP


def get_inter_node_rs_group():
    global FLUX_GEMM_RS_INTER_NODE_GROUP
    assert (
        FLUX_GEMM_RS_INTER_NODE_GROUP is not None
    ), "FLUX_GEMM_RS_INTER_NODE_GROUP not initialized"
    return FLUX_GEMM_RS_INTER_NODE_GROUP


def initialize_inter_node_rs_group(tp_group: dist.ProcessGroup, nnodes: int):
    global FLUX_GEMM_RS_INTER_NODE_GROUP
    if FLUX_GEMM_RS_INTER_NODE_GROUP is not None:
        return

    tp_world_size: int = tp_group.size()
    assert tp_world_size % nnodes == 0, f"{tp_world_size} is not divisible by {nnodes}"
    local_world_size: int = tp_group.size() // nnodes
    world_size: int = dist.get_world_size()
    assert world_size % tp_world_size == 0, f"{world_size} not divisible by {tp_world_size}"
    cur_rank = dist.get_rank()

    for tp0_rank in range(0, world_size, tp_world_size):
        for local_rank in range(local_world_size):
            inter_ranks = [
                tp0_rank + node_rank * local_world_size + local_rank for node_rank in range(nnodes)
            ]
            inter_node_group = dist.new_group(inter_ranks)
            if cur_rank in inter_ranks:
                assert (
                    FLUX_GEMM_RS_INTER_NODE_GROUP is None
                ), "FLUX_GEMM_RS_INTER_NODE_GROUP already initialized"
                FLUX_GEMM_RS_INTER_NODE_GROUP = inter_node_group


class GemmRS_multinode:
    # use multi GemmRS and merge then from python
    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        nnodes: int,
        max_m: int,
        n_dim: int,
        input_dtype: torch.dtype,
        transpose_weight: bool,
        fuse_reduction: bool,
    ) -> None:
        self.tp_group = tp_group
        self.rank = self.tp_group.rank()
        self.world_size = self.tp_group.size()
        self.nnodes = nnodes
        self.local_world_size = self.world_size // self.nnodes
        self.node_id = self.rank // self.local_world_size
        self.local_rank = self.rank % self.local_world_size
        self.max_m_total = max_m
        self.max_m = max_m // self.nnodes
        self.output_buffer = None
        self.cpp_op = None
        self.input_dtype = input_dtype
        self.transpose_weight = transpose_weight
        self.n_dim = n_dim
        self.fuse_reduction = fuse_reduction
        if self.nnodes > 1:
            self.tp_group_intra = get_intra_node_pg_group(tp_group, nnodes)
        else:
            self.tp_group_intra = tp_group
        self.lazy_init_cpp_op(self.max_m)

    def lazy_init_cpp_op(self, m_dim: int):
        if (self.cpp_op is not None) and m_dim <= self.max_m:
            return
        torch.distributed.barrier(self.tp_group_intra)
        self.max_m = m_dim
        self.cpp_op = flux.GemmRS(
            self.tp_group_intra,
            1,
            self.max_m,
            self.n_dim,
            self.input_dtype,
            transpose_weight=self.transpose_weight,
            fuse_reduction=self.fuse_reduction,
        )
        torch.distributed.barrier(self.tp_group_intra)
        torch.cuda.current_stream().synchronize()

    def forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        stream = torch.cuda.current_stream()
        m_dim = input.size(0)
        assert m_dim % self.world_size == 0, f"{m_dim} % {self.world_size} != 0"
        self.lazy_init_cpp_op(m_dim // self.nnodes)
        outputs = []
        for n in range(self.nnodes):
            input_current = input[n * self.max_m : (n + 1) * self.max_m, ...]
            bias_current = bias[n * self.max_m : (n + 1) * self.max_m, ...] if bias else None
            # print(f"gemm rs node {n} with input: {input_current.shape} bias: {bias_current}")
            output = self.cpp_op.forward(input_current, weight, bias_current)
            output_buffer = torch.empty_like(output)
            output_buffer.copy_(output)
            outputs.append(output_buffer)
        # reduce buffer
        recv_buffer = torch.empty_like(outputs[0])
        output = outputs[self.node_id]

        for n in range(1, self.nnodes):
            node_send = (n + self.node_id) % self.nnodes
            rank_send = (n * self.local_world_size + self.rank) % self.world_size
            rank_recv = (self.rank - n * self.local_world_size + self.world_size) % self.world_size
            # print(f"n: {n} node_send: {node_send} rank_send: {rank_send} rank_recv: {rank_recv}")
            reqs = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, outputs[node_send], rank_send, self.tp_group),
                    dist.P2POp(dist.irecv, recv_buffer, rank_recv, self.tp_group),
                ]
            )
            [req.wait() for req in reqs]
            output.add_(recv_buffer)

        return output
