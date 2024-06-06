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

import torch
import torch.distributed as dist
from . import cpp_mod

_INTER_NODE_AG_GEMM_GROUP = None
_INTRA_NODE_AG_GEMM_GROUP = None


def get_inter_node_ag_gemm_group(tp_group: dist.ProcessGroup, nnodes: int):
    global _INTER_NODE_AG_GEMM_GROUP
    if _INTER_NODE_AG_GEMM_GROUP is not None:
        return _INTER_NODE_AG_GEMM_GROUP

    tp_world_size: int = tp_group.size()
    assert tp_world_size % nnodes == 0, f"{tp_world_size} is not divisible by {nnodes}"
    local_world_size: int = tp_group.size() // nnodes
    world_size: int = dist.get_world_size()
    assert world_size % tp_world_size == 0, f"{world_size} not divisible by {tp_world_size}"

    ranks_per_sub_group = []
    for tp0_rank in range(0, world_size, tp_world_size):
        for local_rank in range(local_world_size):
            inter_ranks = [
                tp0_rank + node_rank * local_world_size + local_rank for node_rank in range(nnodes)
            ]
            ranks_per_sub_group.append(inter_ranks)
    _INTER_NODE_AG_GEMM_GROUP, _ = dist.new_subgroups_by_enumeration(ranks_per_sub_group)
    return _INTER_NODE_AG_GEMM_GROUP


def get_intra_node_ag_gemm_group(tp_group: dist.ProcessGroup, nnodes: int):
    global _INTRA_NODE_AG_GEMM_GROUP
    if _INTRA_NODE_AG_GEMM_GROUP is not None:
        return _INTRA_NODE_AG_GEMM_GROUP

    tp_world_size: int = tp_group.size()
    assert tp_world_size % nnodes == 0, f"{tp_world_size} is not divisible by {nnodes}"
    local_world_size: int = tp_group.size() // nnodes
    world_size: int = dist.get_world_size()
    assert world_size % tp_world_size == 0, f"{world_size} not divisible by {tp_world_size}"

    ranks_per_sub_group = []
    for node_id in range(nnodes):
        ranks_per_sub_group.append(
            list(range(node_id * local_world_size, (node_id + 1) * local_world_size))
        )
    _INTRA_NODE_AG_GEMM_GROUP, _ = dist.new_subgroups_by_enumeration(ranks_per_sub_group)
    return _INTRA_NODE_AG_GEMM_GROUP


class AGGemm0:
    def __init__(
        self,
        weight: torch.Tensor,
        tp_group,
        full_m: int,
        nnodes: int = 1,
        transpose_weight: bool = False,
        gather_output=False,
    ) -> None:
        self.weight = weight
        self.tp_group = tp_group
        self.rank = self.tp_group.rank()
        self.world_size = self.tp_group.size()
        self.full_m = full_m
        self.nnodes = nnodes
        assert self.world_size % nnodes == 0
        assert self.full_m % self.world_size == 0
        self.local_world_size = self.world_size // nnodes
        self.local_rank = self.rank % self.local_world_size
        if nnodes > 1:
            self.inter_node_group = get_inter_node_ag_gemm_group(tp_group, nnodes)
        self.intra_node_group = get_intra_node_ag_gemm_group(tp_group, nnodes)

        self.transpose_weight = transpose_weight
        self.gather_output = gather_output
        self.n_dim = weight.size(1) if transpose_weight else weight.size(0)
        self.k_dim = weight.size(0) if transpose_weight else weight.size(1)

        # input buffer to hold data from other devices
        self.input_buffer = None
        self.shm_handles = None
        self.shm_offsets = None

        self.output_buffer = None

        # signal for dependency
        self.signal = None
        self.signal_handles = None
        self.signal_offsets = None

        self.cpp_op = None
        self.lazy_init_cpp_op(self.full_m)

    def lazy_init_cpp_op(self, full_m: int):
        if self.cpp_op is not None:
            return

        self.full_m = full_m

        # input buffer and handle
        self.input_buffer = torch.zeros(
            (full_m, self.k_dim),
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False,
        )
        input_buffer_offset = self.input_buffer.storage_offset()
        shm_handle = self.input_buffer._typed_storage()._share_cuda_()[1]
        shm_offset = self.input_buffer._typed_storage()._share_cuda_()[3]
        shm_handle_ts_cuda = torch.ByteTensor(torch.ByteStorage._from_buffer(shm_handle)).cuda()
        shm_handles = [torch.empty_like(shm_handle_ts_cuda) for _ in range(self.local_world_size)]
        torch.distributed.all_gather(shm_handles, shm_handle_ts_cuda, group=self.intra_node_group)
        self.shm_handles = [handle.cpu() for handle in shm_handles]

        offset_value = shm_offset + input_buffer_offset
        offset_list = [None for _ in range(self.local_world_size)]
        torch.distributed.all_gather_object(offset_list, offset_value, group=self.intra_node_group)
        self.shm_offsets = offset_list

        # output buffer
        self.output_buffer = torch.zeros(
            (full_m, self.n_dim),
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False,
        )

        # cuda signal
        num_stages = self.local_world_size
        self.signal = torch.zeros(
            (num_stages),
            dtype=torch.int64,
            device=self.weight.device,
            requires_grad=False,
        )
        signal_tensor_offset = self.signal.storage_offset()

        signal_handle = self.signal._typed_storage()._share_cuda_()[1]
        signal_offset = self.signal._typed_storage()._share_cuda_()[3]
        signal_handle_ts_cuda = torch.ByteTensor(
            torch.ByteStorage._from_buffer(signal_handle)
        ).cuda()
        signal_handles = [
            torch.empty_like(signal_handle_ts_cuda) for _ in range(self.local_world_size)
        ]
        torch.distributed.all_gather(
            signal_handles, signal_handle_ts_cuda, group=self.intra_node_group
        )
        self.signal_handles = [handle.cpu() for handle in signal_handles]

        signal_total_offset = signal_tensor_offset + signal_offset
        signal_offset_list = [None for _ in range(self.local_world_size)]
        torch.distributed.all_gather_object(
            signal_offset_list, signal_total_offset, group=self.intra_node_group
        )
        self.signal_offsets = signal_offset_list

        self.cpp_op = cpp_mod.AllGatherGemm(
            self.local_rank,
            self.local_world_size,
            self.input_buffer,
            self.weight,
            self.output_buffer,
            self.shm_handles,
            self.shm_offsets,
            self.signal,
            self.signal_handles,
            self.signal_offsets,
            self.transpose_weight,
            self.gather_output,
        )

    def _lazy_init_inter_ag_output_buffer(self, input: torch.Tensor):
        input_shape = input.size()
        m_dim, k_dim = input_shape[-2], input_shape[-1]
        self.inter_ag_output_buffer = torch.empty(
            (m_dim * self.nnodes, k_dim),
            dtype=input.dtype,
            device=input.device,
            requires_grad=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.nnodes > 1:
            self._lazy_init_inter_ag_output_buffer(input)
            dist.all_gather_into_tensor(
                self.inter_ag_output_buffer, input, group=self.inter_node_group
            )
            input = self.inter_ag_output_buffer

        self.lazy_init_cpp_op(self.full_m)
        full_output = self.cpp_op.forward(input)
        gather_out = self.input_buffer if self.gather_output else None
        return full_output, gather_out
