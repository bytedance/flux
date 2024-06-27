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

MAX_NUM_SIGNAL = 64
INTER_NODE_AG_KERNEL_GROUP = None
INTRA_NODE_AG_KERNEL_GROUP = None


def get_inter_node_ag_group(tp_group: dist.ProcessGroup, nnodes: int):
    global INTER_NODE_AG_KERNEL_GROUP
    if INTER_NODE_AG_KERNEL_GROUP is not None:
        return INTER_NODE_AG_KERNEL_GROUP
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

    INTER_NODE_AG_KERNEL_GROUP, _ = dist.new_subgroups_by_enumeration(ranks_per_sub_group)
    return INTER_NODE_AG_KERNEL_GROUP


def get_intra_node_ag_group(tp_group: dist.ProcessGroup, nnodes: int):
    global INTRA_NODE_AG_KERNEL_GROUP
    if INTRA_NODE_AG_KERNEL_GROUP is not None:
        return INTRA_NODE_AG_KERNEL_GROUP

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

    INTRA_NODE_AG_KERNEL_GROUP, _ = dist.new_subgroups_by_enumeration(ranks_per_sub_group)
    return INTRA_NODE_AG_KERNEL_GROUP


def _auto_ring_mode() -> cpp_mod.AgRingMode:
    import os

    if os.getenv("FLUX_AG_CROSSNODE_RING_MODE") is not None:
        return cpp_mod.AgRingMode(os.getenv("FLUX_AG_CROSSNODE_RING_MODE"))
    force_nvlink = int(os.getenv("FLUX_FORCE_NVLINK", "-1"))
    if force_nvlink == 1:  # nvlink mode
        return cpp_mod.AgRingMode.All2All
    elif force_nvlink == 0:  # pci-e mode
        return cpp_mod.AgRingMode.Ring2D

    return cpp_mod.AgRingMode.Auto


class AGKernelXNode:
    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        nnodes: int,
        full_m: int,
        n_dim: int,
        k_dim: int,
        input_dtype: torch.dtype,
        transpose_weight: bool = True,
        gather_output: bool = False,
        local_copy: bool = False,
        ring_mode: cpp_mod.AgRingMode = cpp_mod.AgRingMode.Auto,
    ) -> None:
        """
        : ring_mode[int]
            -1 for auto: nvlink machine default to 0, non-nvlink machine default to 2
            0: nvlink mode: all-to-all
            1: 1d ring. for PCI-e communication optimization
            2: 2d ring. for PCI-e communication optimization. better performance than 1d ring
            3: custom ring. for defining arbitrary ring at compile time
        """
        self.tp_group = tp_group
        self.world_size = self.tp_group.size()
        self.local_world_size = self.world_size // nnodes
        self.rank = self.tp_group.rank()
        self.local_rank = self.rank % self.local_world_size
        self.nnodes = nnodes

        self.cross_node = self.nnodes > 1
        if self.cross_node:
            self.inter_node_group = get_inter_node_ag_group(tp_group, nnodes)
        self.intra_node_group = get_intra_node_ag_group(tp_group, nnodes)

        self.input_dtype = input_dtype
        self.full_m = full_m
        self.n_dim = n_dim
        self.k_dim = k_dim
        self.transpose_weight = transpose_weight
        self.gather_output = gather_output
        self.local_copy = local_copy
        # input buffer
        self.input_buffer = None
        # output buffer
        self.output_buffer = None
        if ring_mode == cpp_mod.AgRingMode.Auto:
            self.ring_mode = _auto_ring_mode()
        else:
            self.ring_mode = ring_mode
        print(f"ring_mode: {self.ring_mode}")

        self.cpp_op = None
        self.lazy_init_cpp_op(self.full_m)

    def lazy_init_cpp_op(self, full_m: int):
        if self.cpp_op is not None:
            return

        self.full_m = full_m

        # output buffer
        self.output_buffer = torch.zeros(
            (full_m, self.n_dim),
            dtype=self.input_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        self.cpp_op = cpp_mod.AGKernelCrossNode(
            self.tp_group,
            self.intra_node_group,
            self.nnodes,
            self.output_buffer,
            self.full_m,
            self.n_dim,
            self.k_dim,
            self.input_dtype,
            self.transpose_weight,
            self.local_copy,
            self.ring_mode,
        )

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        self.lazy_init_cpp_op(self.full_m)
        output = self.cpp_op.forward(input, weight)
        gather_output = self.input_buffer if self.gather_output else None
        return output, gather_output

    def reset_signals(self):
        self.cpp_op.reset_signals()

    def copy_local(self, input: torch.Tensor):
        self.cpp_op.copy_local(input)

    def gemm_only(
        self, input: torch.Tensor, full_input: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        output = self.cpp_op.gemm_only(input, full_input, weight)
        return output


__all__ = ["AGKernelXNode"]
