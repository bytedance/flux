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
from typing import Optional, List

def bsr_reduce(input: torch.Tensor, output: torch.Tensor, block_h: int, block_w: int): ...
def bitwise_check(A: torch.Tensor, B: torch.Tensor) -> bool: ...
def uniform_initialize(tensor: torch.Tensor, seed: int, min: float, max: float) -> bool: ...

class TuningRecord:
    pass

def load_tuning_record(record: TuningRecord) -> None: ...

class ProfilingContext:
    def __init__(self, name: str): ...
    def get_code(self) -> str: ...
    def get_latest_prof_result(self) -> str: ...
    def get_all_prof_results(self) -> List[str]: ...
    def get_latest_record(self) -> TuningRecord: ...
    def get_all_records(self) -> List[TuningRecord]: ...

class DistEnvTP:
    def __init__(tp_group: dist.ProcessGroup, nnodes: int) -> None: ...

class DistEnvTPWithEP:
    def __init__(
        tp_group: dist.ProcessGroup, nnodes: int, ep_group: Optional[dist.ProcessGroup] = None
    ) -> None: ...

class GemmOnly:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        transpose_weight: bool = False,
    ): ...
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output_buf: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
    ) -> torch.Tensor: ...
    def profiling(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output_buf: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        prof_ctx: Optional[ProfilingContext] = None,
    ) -> torch.Tensor: ...

class GemmRS:
    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        nnodes: int,
        max_m: int,
        n_dim: int,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        transpose_weight=False,
        fuse_reduction=False,
    ): ...
    def forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...
    def forward_gemm(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None: ...
    def forward_barrier(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None: ...
    def forward_reduce_scatter(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...
    def zero_buffers(self) -> None: ...
    def profiling(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        prof_ctx: Optional[ProfilingContext] = None,
    ) -> torch.Tensor: ...

class GemmRSAcrossNode:
    def __init__(
        self,
        rank: int,
        world_size: int,
        local_world_size: int,
        weight: torch.Tensor,
        transpose_weight: bool,
        output_buffer: torch.Tensor,
        output: torch.Tensor,
        barrier: torch.Tensor,
        fuse_reduction: bool,
    ): ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

class AllGatherGemm:
    def __init__(
        self,
        rank: int,
        world_size: int,
        input_buffer: torch.Tensor,
        weight: torch.Tensor,
        output_buffer: torch.Tensor,
        cuda_shm_handles: List[torch.Tensor],
        signal: torch.Tensor,
        signal_handles: List[torch.Tensor],
        signal_offsets: List[int],
        transpose_weight: bool,
        gather_output: bool,
    ): ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

class AGKernel:
    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        nnodes: int,
        full_m: int,
        n_dim: int,
        k_dim: int,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        transpose_weight: bool = True,
        local_copy: bool = True,
        ring_mode: int = -1,
    ): ...
    def forward_allgather(self, input: torch.Tensor) -> None: ...
    def forward_gemm(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...
    def forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...
    def gather(self) -> torch.Tensor: ...
    def reset_signals(self) -> None: ...
    def copy_local(self, input: torch.Tensor) -> None: ...
    def profiling(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        prof_ctx: Optional[ProfilingContext] = None,
    ) -> torch.Tensor: ...

class AGKernelCrossNode:
    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        nnodes: int,
        output_buffer: torch.Tensor,
        full_m: int,
        n_dim: int,
        k_dim: int,
        input_dtype: torch.dtype,
        transpose_weight: bool = True,
        local_copy: bool = False,
        ring_mode: int = -1,  # 0 for nvlink, 2 for 2d ring. 1 for 1d ring. -1 for auto selection: 0 for NVLINK machine and 2 for PCI-e machine
    ): ...
    def reset_signals(self) -> None: ...
    def copy_local(self, input: torch.Tensor) -> None: ...
    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
    def gemm_only(
        self, input: torch.Tensor, full_input: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor: ...
