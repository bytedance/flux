from enum import Enum
import torch
import torch.distributed as dist
from typing import Optional, List

def bsr_reduce(input: torch.Tensor, output: torch.Tensor, block_h: int, block_w: int): ...
def bitwise_check(A: torch.Tensor, B: torch.Tensor) -> bool: ...
def uniform_initialize(tensor: torch.Tensor, seed: int, min: float, max: float) -> bool: ...

class TuningRecord:
    pass

class AgRingMode(Enum):
    All2All = ...
    Ring1D = ...
    Ring2D = ...
    Auto = ...

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

class MoeArguments:
    def __init__(
        max_ntokens: int,
        hidden: int,
        ffn_hidden: int,
        nexperts: int,
        topk: int,
        input_dtype=torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
    ) -> None: ...

class GemmOnly:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        transpose_weight: bool = False,
        use_fp8_gemm: bool = False,
    ): ...
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output_buf: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
    ) -> torch.Tensor: ...
    def profiling(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output_buf: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
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
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
    ) -> torch.Tensor: ...
    def forward_gemm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
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
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
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
        ring_mode: AgRingMode = AgRingMode.Auto,
    ): ...
    def forward_allgather(self, input: torch.Tensor) -> None: ...
    def forward_gemm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
    ) -> torch.Tensor: ...
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
    ) -> torch.Tensor: ...
    def gather(self) -> torch.Tensor: ...
    def reset_signals(self) -> None: ...
    def copy_local(self, input: torch.Tensor) -> None: ...
    def profiling(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
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
        ring_mode: AgRingMode = AgRingMode.Auto,
    ): ...
    def reset_signals(self) -> None: ...
    def copy_local(self, input: torch.Tensor) -> None: ...
    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
    def gemm_only(
        self, input: torch.Tensor, full_input: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor: ...

class All2AllOp:
    def __init__(self, rank: int, world_size: int, recv_buffer: torch.Tensor): ...
    def forward(self, send_buffer: List[torch.Tensor]) -> None: ...

class GemmGrouped:
    def __init__(self, weight: torch.Tensor, num_experts: int): ...
    def forward(self, input: torch.Tensor, split_cpu: torch.Tensor) -> torch.Tensor: ...

class GemmGroupedV3:
    def __init__(self, weight: torch.Tensor, num_experts: int): ...
    def forward(self, input: torch.Tensor, split_cpu: torch.Tensor) -> torch.Tensor: ...

class GemmGroupedV3AGScatter:
    def __init__(self, tp_env: DistEnvTP, moe_args: MoeArguments): ...
    def forward(
        self,
        inputs_shard: torch.Tensor,
        weights: torch.Tensor,
        splits_gpu: torch.Tensor,
        scatter_index: torch.Tensor,
        output_scale: Optional[torch.Tensor] = None,
        outputs_buf: Optional[torch.Tensor] = None,
        allgather_output: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        sm_mrgin: int = 0,
    ) -> torch.Tensor: ...
    def forward_multiple_weights(
        self,
        inputs_shard: torch.Tensor,
        weights: List[torch.Tensor],
        splits_gpu: torch.Tensor,
        scatter_index: torch.Tensor,
        output_scale: Optional[List[torch.Tensor]] = None,
        outputs_buf: Optional[List[torch.Tensor]] = None,
        allgather_output: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        sm_mrgin: int = 0,
    ) -> List[torch.Tensor]: ...
    def clear_buffers(self) -> None: ...
    def profiling(
        self,
        inputs_shard: torch.Tensor,
        weights: List[torch.Tensor],
        splits_gpu: torch.Tensor,
        scatter_index: torch.Tensor,
        output_scale: Optional[List[torch.Tensor]] = None,
        outputs_buf: Optional[List[torch.Tensor]] = None,
        allgather_output: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        sm_mrgin: int = 0,
        prof_ctx: Optional[ProfilingContext] = None,
    ) -> List[torch.Tensor]: ...

class GemmGroupedV3GatherRS:
    def __init__(
        self,
        num_experts: int,
        max_m: int,
        n_dim: int,
        topk: int,
        rank: int,
        world_size: int,
        tp_world_size: int,
        ep_world_size: int,
    ): ...
    def forward_gather_rs(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        split_cpu: torch.Tensor,
        scatter_idx: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fastacc: bool = True,
        sm_mrgin: int = 0,
        with_stream_sync: bool = False,
    ) -> torch.Tensor: ...
    def forward_gather_rs_multiple(
        self,
        input: List[torch.Tensor],
        weight: List[torch.Tensor],
        split_cpu: torch.Tensor,
        scatter_idx: torch.Tensor,
        input_scale: Optional[List[torch.Tensor]] = None,
        weight_scale: Optional[List[torch.Tensor]] = None,
        output_scale: Optional[List[torch.Tensor]] = None,
        fastacc: bool = True,
        sm_mrgin: int = 0,
        with_stream_sync: bool = False,
    ) -> torch.Tensor: ...
