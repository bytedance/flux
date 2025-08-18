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

import ctypes
from enum import Enum
from typing import List, Optional, Tuple
import typing

import torch
import torch.distributed as dist
import torch.distributed

def bsr_reduce(input: torch.Tensor, output: torch.Tensor, block_h: int, block_w: int): ...
def bitwise_check(A: torch.Tensor, B: torch.Tensor) -> bool: ...
def uniform_initialize(tensor: torch.Tensor, seed: int, min: float, max: float) -> bool: ...
def init_flux_shm(pg: dist.ProcessGroup) -> None: ...
def create_tensor(shape: List[int], dtype: torch.dtype, pg: dist.ProcessGroup) -> torch.Tensor: ...
def create_tensor_list(
    shape: List[int], dtype: torch.dtype, pg: dist.ProcessGroup
) -> List[torch.Tensor]: ...
def flux_create_shm_tensor_list(
    shape: List[int], dtype: torch.dtype, pg: dist.ProcessGroup
) -> torch.Tensor: ...
def topk_scatter_reduce(
    inputs: List[torch.Tensor], scatter_index: torch.Tensor, topk: int
) -> torch.Tensor: ...

class SegmentInfo:
    @property
    def segment_origin(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def tile_m_start_new(self) -> int: ...
    @property
    def tile_m_start_origin(self) -> int: ...

def get_gemm_rs_threadblock_segments_info(
    problem_shape: List[int],
    tiled_shape: List[int],
    rank: int,
    world_size: int,
    sub_world_size: int,
    nnodes: int,
    use_2d_ring: bool,
    per_tile_flags: bool,
) -> list[SegmentInfo]: ...
def calc_gemm_rs_threadblock_segments_info(
    segments: torch.Tensor,
    problem_shape: List[int],
    tiled_shape: List[int],
    rank: int,
    world_size: int,
    sub_world_size: int,
    nnodes: int,
    use_2d_ring: bool,
    per_tile_flags: bool,
): ...

class GroupBarrier:
    def __init__(self, pg: torch.distributed.ProcessGroup, ring_mode: bool) -> None:
        """
        group_name: torch.distributed.ProcessGroup.group_name
        ring_mode: use ring_mode barrier or not. force use ring_mode=True if torch.cuda.device_count() > 8
            use NVSHMEM implementation if compiled with --nvshmem and ring_mode=False, use flux implementation if not.
        """

    def barrier_all(self, stream: int) -> None: ...

class TuningRecord:
    pass

class AGRingMode:
    """
    Members:

      All2All

      Ring1D

      Ring2D
    """

    All2All: typing.ClassVar[AGRingMode]  # value = <AGRingMode.All2All: 0>
    Ring1D: typing.ClassVar[AGRingMode]  # value = <AGRingMode.Ring1D: 1>
    Ring2D: typing.ClassVar[AGRingMode]  # value = <AGRingMode.Ring2D: 2>
    __members__: typing.ClassVar[
        dict[str, AGRingMode]
    ]  # value = {'All2All': <AGRingMode.All2All: 0>, 'Ring1D': <AGRingMode.Ring1D: 1>, 'Ring2D': <AGRingMode.Ring2D: 2>}


class AllGatherOption:
    """
    TODO(houqi.1993)
        * for small input, use CUDA core implementation.

    input_buffer_copied: if input and input_scale is ready. default to False. If True,
    use_cuda_core_local: default to True for INT8 input with input_scale. otherwise False
    use_cuda_core_ag: default to True for INT8 input with input_scale with small shape. otherwise False
    fuse_sync: fuse sync into copy function.
    use_read: read for pull mode, write for push mode. for A100 NVLink, read mode is better or faster. otherwise write mode is better.
    mode: use AlltoAll mode or ring mode(Ring1D or Ring2D)
    """

    def __init__(self) -> None: ...
    input_buffer_copied: Optional[bool]
    use_cuda_core_local: Optional[bool]
    use_cuda_core_ag: Optional[bool]
    fuse_sync: Optional[bool]
    use_read: Optional[bool]
    mode: Optional[AGRingMode]

class RingMode:
    All2All: typing.ClassVar[RingMode]  # value = <RingMode.All2All: 0>
    Ring1D: typing.ClassVar[RingMode]  # value = <RingMode.Ring1D: 1>
    Ring2D: typing.ClassVar[RingMode]  # value = <RingMode.Ring2D: 2>
    __members__: typing.ClassVar[
        dict[str, RingMode]
    ]  # value = {'All2All': <RingMode.All2All: 0>, 'Ring1D': <RingMode.Ring1D: 1>, 'Ring2D': <RingMode.Ring2D: 2>}

class ReduceScatterOption:
    use_barrier_queue: Optional[bool]
    use_1d_ring: Optional[bool]
    use_p2p_read: Optional[bool]
    use_cudaMemcpyAsync: Optional[bool]
    use_gemmk: Optional[bool]
    per_tile_flags: Optional[bool]
    n_split: Optional[int]
    num_blocks: Optional[int]
    ring_mode: Optional[RingMode]

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
    """
    support 4 mode: FP16/BF16 mode, FP8 mode, INT8 Dequant mode, INT8 (GEMM)Only mode

    About shapes:
    * input: [M, K] for all types
    * weight: [K, N] if transpose_weight, [N, K] if not transpose_weight.
    * bias: [1, N] for FP8 or INT8 Dequant, [M, N] for FP16/BF16/INT8(Only).
    * input_scale: always None for FP16/BF16/Int8 Only. [M, 1] for INT8 Dequant. [1] for FP8
    * weight_scale: always None for FP16/BF16/Int8 Only. [1, N] for INT8 Dequant. [1] for FP8
    * output_scale: always None for FP16/BF16/INT8 Dequant/INT8 Only. for FP8 ???

    for transpose_weight=True:
        for FP16:  output_FP16 = [input_FP16 * weight_FP16]_FP32.to(FP16) + bias_FP16
        for BF16:  output_BF16 = [input_BF16 * weight_BF16]_FP32.to(BF16) + bias_BF16
        for FP8:   output_BF16 = [[input_FP8 * weight_FP8]_FP32 * input_scale_FP32 * weight_scale_FP32]_FP32.to(BF16) + bias_FP16
        for INT8 Dequant:  output_BF16 = [[input_INT8 * weight_INT8]_INT32.to(FP32) * input_scale_FP32 * weight_scale_FP32]_FP32.to(BF16) + bias_BF16
        for INT8 Only: output_INT32 = [input_INT8 * weight_INT8]_INT32 + bias_INT32
    for transpose_weight=False, replace weight with weight.T
    """

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

class BlockScaleGemm:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
    ): ...
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...
    def forward_multi_stream(
        self,
        input: torch.Tensor,
        input_list: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...
    def reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...
    def profiling(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
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
        ring_reduction=False,
    ):
        """The formula:
        * matmul(input, weight) * input_scale * weight_scale + bias   if transpose_weight=True
        * matmul(input, weight.T) * input_scale * weight_scale + bias   if transpose_weight=False
        """

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        reduce_scatter_option: ReduceScatterOption = ReduceScatterOption(),
    ) -> torch.Tensor:
        """
        support 3 mode: FP16/BF16 mode, FP8 mode, INT8 mode

        TODO(wenlei.bao)
        * what about output_scale?

        About shapes: M = M_per_rank * world_size
        * input: [M_per_rank, K] for all types
        * weight: [K, N] if transpose_weight, [N, K] if not transpose_weight.
        * bias: [1, N] for FP8 or INT8, [M_per_rank, N] for FP16 or BF16.
        * input_scale: always None for FP16/BF16. [M, 1] for INT8. [1] for FP8
        * weight_scale: always None for FP16/BF16. [1, N] for INT8. [1] for FP8
        * output_scale: always None for FP16/BF16/INT8. for FP8 ???

        for transpose_weight=True:
            for FP16:  output_FP16 = [input_FP16 * weight_FP16]_FP32.to(FP16) + bias_FP16
            for BF16:  output_BF16 = [input_BF16 * weight_BF16]_FP32.to(BF16) + bias_BF16
            for FP8:   output_BF16 = [[input_FP8 * weight_FP8]_FP32 * input_scale_FP32 * weight_scale_FP32]_FP32.to(BF16) + bias_FP16
            for INT8:  output_BF16 = [[input_INT8 * weight_INT8]_INT32.to(FP32) * input_scale_FP32 * weight_scale_FP32]_FP32.to(BF16) + bias_BF16
        for transpose_weight=False, replace weight with weight.T
        """

    def forward_barrier(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None: ...
    def forward_reduce_scatter(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
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
    ): ...
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        transpose_weight: bool = False,
        all_gather_option: AllGatherOption = AllGatherOption(),
        gathered_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        support 3 mode: BF16/FP16 mode or FP8 mode or int8 mode

        TODO(wenlei.bao)
        * what about output_scale?

        About shapes: M = M_per_rank * world_size
        * input: [M_per_rank, K] for all types
        * weight: [K, N] if transpose_weight, [N, K] if not transpose_weight.
        * bias: [1, N] for FP8 or INT8, [M_per_rank, N] for FP16 or BF16.
        * output: [M, N]
        * input_scale: [M_per_rank, 1] for INT8, [1] for FP8.
        * weight_scale: [1, N] for INT8, [1] for FP8
        * output_scale: ?

        About Optional:
        * output: optional for all modes. if set, the output will be written to this buffer.
        * bias: optional for all modes. zero for None.
        * input_scale/weight_scale: always None for FP16/BF16. always not None for INT8. for FP8 ??
        * output_scale: always None for FP16/BF16/INT8. for FP8 ???
        * gathered_input: if not set to None, the input will be allgathered to this tensor.

        About dtypes and formula:
        * for FP16 mode:  output_FP16 = [input_FP16 * weight_FP16]_FP32.to(FP16) + bias_FP16
        * for BF16 mode:  output_BF16 = [input_BF16 * weight_BF16]_FP32.to(BF16) + bias_BF16
        * for FP8  mode:  output_BF16 = [[input_FP8 * weight_FP8]_FP32.to(BF16) * input_scale_FP32 * weight_scale_FP32]_FP32.to(BF16) + bias_BF16
        * for int8 mode:  output_BF16 = [[input_INT8 * weight_INT8]_INT32.to(FP32) * input_scale_FP32 * weight_scale_FP32]_FP32.to(BF16) + bias_BF16
        """

    def gemm_only(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        transpose_weight: bool = False,
    ) -> torch.Tensor: ...
    def profiling(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        transpose_weight: bool = False,
        all_gather_option: AllGatherOption = AllGatherOption(),
        gathered_input: Optional[torch.Tensor] = None,
        prof_ctx: Optional[ProfilingContext] = None,
    ) -> torch.Tensor: ...

class AGKernelCrossNode:
    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        intra_node_group: torch.distributed.ProcessGroup,
        nnodes: int,
        output_buffer: torch.Tensor,
        full_m: int,
        n_dim: int,
        k_dim: int,
        input_dtype: torch.dtype,
        transpose_weight: bool = True,
        local_copy: bool = False,
        ring_mode: AGRingMode | None = None,
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

class GemmGroupedV2:
    def __init__(
        self, weight: torch.Tensor, num_experts: int, in_type: torch.dtype, out_type: torch.dtype
    ): ...
    def forward(
        self,
        input: torch.Tensor,
        split_cpu: torch.Tensor,
        input_scale: Optional[List[torch.Tensor]] = None,
        weight_scale: Optional[List[torch.Tensor]] = None,
        fast_accum: bool = False,
        sm_margin: int = 0,
    ) -> torch.Tensor: ...

class GemmGroupedV3:
    def __init__(self, weight: torch.Tensor, num_experts: int): ...
    def forward(self, input: torch.Tensor, split_cpu: torch.Tensor) -> torch.Tensor: ...
    def profiling(
        self, input: torch.Tensor, splits_cpu: torch.Tensor, prof_ctx: ProfilingContext = None
    ) -> torch.Tensor: ...

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
        sm_margin: int = 0,
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
        sm_margin: int = 0,
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
        sm_margin: int = 0,
        prof_ctx: Optional[ProfilingContext] = None,
    ) -> List[torch.Tensor]: ...

class GemmGroupedV2AGScatterOp:
    def __init__(
        self,
        tp_env: DistEnvTP,
        moe_args: MoeArguments,
    ): ...
    def forward(
        self,
        inputs_shard: torch.Tensor,
        weights: torch.Tensor,
        splits_gpu: torch.Tensor,
        scatter_index: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        outputs_buf: Optional[torch.Tensor] = None,
        allgather_output: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        sm_margin: int = 0,
        ag_option: AllGatherOption = AllGatherOption(),
    ) -> torch.Tensor: ...
    """
    2 modes supported: FP16/BF16, or FP8 mode.  INT8 is not supported yet.

    About some variables:
        * ntokens = tokens_per_rank * world_size.
        * M = ntokens * topk.
        * sum(M_this_ep) = M. for EP=1, M_this_ep = M.

    About shapes:
    * inputs_shard: [M_this_ep, K] for all types
    * weights: [E, K, N] if transpose_weight, [E, N, K] if not transpose_weight.
    * splits_gpu: [E] of torch.int32 by default or [E+1] if drop_token. for EP=1, M_this_rank = M * topk.
    * scatter_index: [ntokens, topk] of torch.int32
    * output_scale: [E] of torch.float32
    * output_buf: [M_this_ep, N].
    * allgather_output: [ntokens, K] of the same dtype as inputs_shard

    input_{T} = index_select(all_gather(inputs_shard_{T}), dim=0, index=gather_index)  for T in [FP16,BF16,FP8]
    for FP16 mode: output_FP16 = [input_FP16 * weights_FP16]_FP32.to(FP16)
    for BF16 mode: output_BF16 = [input_BF16 * weights_BF16]_FP32.to(BF16)
    for FP8  mode: output_BF16 = [[input_FP8 * weights_FP8]_FP32 * output_scale_FP32]_FP32.to(BF16)

    """

    def forward_multiple_weights(
        self,
        inputs_shard: torch.Tensor,
        weights: List[torch.Tensor],
        splits_gpu: torch.Tensor,
        scatter_index: torch.Tensor,
        input_scale: Optional[List[torch.Tensor]] = None,
        weight_scale: Optional[List[torch.Tensor]] = None,
        output_scale: Optional[List[torch.Tensor]] = None,
        outputs_buf: Optional[List[torch.Tensor]] = None,
        allgather_output: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        sm_margin: int = 0,
        ag_option: AllGatherOption = AllGatherOption(),
    ) -> List[torch.Tensor]: ...
    def clear_buffers(self) -> None: ...
    def profiling(
        self,
        inputs_shard: torch.Tensor,
        weights: List[torch.Tensor],
        splits_gpu: torch.Tensor,
        scatter_index: torch.Tensor,
        input_scale: Optional[List[torch.Tensor]] = None,
        weight_scale: Optional[List[torch.Tensor]] = None,
        output_scale: Optional[List[torch.Tensor]] = None,
        outputs_buf: Optional[List[torch.Tensor]] = None,
        allgather_output: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        sm_margin: int = 0,
        ag_option: AllGatherOption = AllGatherOption(),
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
        sm_margin: int = 0,
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
        sm_margin: int = 0,
        with_stream_sync: bool = False,
    ) -> torch.Tensor: ...

class GemmGroupedV2GatherRSOp:
    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        total_num_experts: int,
        max_m: int,
        n_dim: int,
        topk: int,
        output_dtype: torch.dtype,
        tp_world_size: int,
        ep_world_size: int,
        max_input_groups: int = 1,
        n_split: int = 4,
        do_all_reduce: bool = False,
        use_read_mode: bool = False,
    ):
        """
        such conditions expected:
            max_input_groups <= 2
            tp_world_size * ep_world_size == tp_group.size
        """
        ...

    def forward_gather_rs(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        splits_cpu: torch.Tensor,
        scatter_idx: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_vec_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = True,
        sm_margin: int = 0,
        with_stream_sync: bool = False,  # NOTE: not used. to align with V3
    ) -> torch.Tensor:
        """
        support 3 modes: FP16/BF16 mode, or FP8(FP8E4M3FN/FP8E5M2) mode, or INT8 mode.

        TODO(houqi.1993) INT8 mode is not supported

        some params:
        * E_this_ep: sum(E_this_ep) = E. for EP=1, E_this_ep = E.
        * M_this_ep: sum(M_this_ep) = M. for EP=1, M_this_ep = M = ntokens * topk

        some shapes:
        * input: [M_this_ep, K] for all types
        * weight: [E_this_ep, K, N] if transpose_weight, [E_this_ep, N, K] if not transpose_weight.
        * bias: [N] for int8
        * input_scale: [1] for FP16/BF16/FP8, [M_this_ep] for INT8 dynamic quant
        * weight_scale: [E_this_ep, 1] for FP16/BF16/FP8, [E_this_ep, N] for INT8.

        for FP16:
            gemm_out_FP16(m_indexes) =
                [input(m_indexes(i), :)_FP16 * weight(i, :, :)_FP16]_FP32
                * input_scale(None, None)_FP32
                * weight_scale(i, None, None)_FP32
                * output_scale(m_indexes(i), None)_FP32]_FP32.to(FP16)  . for expert i in E_this_ep.
            scatter_idx -> gather_index
            output_FP16 = reduce_scatter(select_index(gemm_out_FP16, gather_index)_FP16)_FP16
        for BF16: replace FP16 with BF16 and use FP16 formula.
        for FP8(both FP8E4M3FN or FP8E5M2):
            gemm_out_BF16(m_indexes) =
                [input(m_indexes(i), :)_FP8 * weight(i, :, :)_FP8]_FP32
                * input_scale(None, None)_FP32
                * weight_scale(i, None, None)_FP32
                * output_scale(m_indexes(i), None)_FP32]_FP32.to(BF16)  . for expert i in E_this_ep.
            output_BF16 = reduce_scatter(select_index(gemm_out_BF16, gather_index)_BF16)_BF16
        for INT8:
            gemm_out_BF16(m_indexes) =
                [input(m_indexes(i), :)_INT8 * weight(i, :, :)_INT8]_FP32
                * weight_scale(i, None, :)_FP32
                * output_scale(m_indexes(i), None)_FP32]_FP32.to(BF16)
                 + bias . for expert i in E_this_ep.
            output_BF16 = reduce_scatter(select_index(gemm_out_BF16, gather_index)_BF16)_BF16
        """

    def forward_gather_rs_multiple(
        self,
        input: List[torch.Tensor],
        weight: List[torch.Tensor],
        splits_cpu: torch.Tensor,
        scatter_idx: torch.Tensor,
        input_scale: Optional[List[torch.Tensor]] = None,
        weight_scale: Optional[List[torch.Tensor]] = None,
        output_vec_scale: Optional[List[torch.Tensor]] = None,
        fast_accum: bool = True,
        sm_margin: int = 0,
        with_stream_sync: bool = False,
    ) -> torch.Tensor: ...
    def profiling(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        splits_cpu: torch.Tensor,
        scatter_idx: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_vec_scale: Optional[torch.Tensor] = None,
        fastacc: bool = True,
        sm_margin: int = 0,
        with_stream_sync: bool = False,
        prof_ctx: Optional[ProfilingContext] = None,
    ) -> torch.Tensor: ...

class TopkReduceScatterOp:
    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        max_m: int,
        n_dim: int,
        topk: int,
        output_dtype: torch.dtype,
        num_experts: int,
        ep_world_size: int,
        barriers: List[torch.Tensor],
        n_split: int = 4,
        do_all_reduce: bool = False,
        use_read_mode: bool = False,
    ): ...
    def run(
        self,
        group_gemm_buffers: List[torch.Tensor],
        output: Optional[torch.Tensor],
        ep_start: int,
        ep_nexperts: int,
        splits: torch.Tensor,
        routing_idx: torch.Tensor,
        output_vec_scales: List[torch.Tensor],
        num_thread_blocks: int,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor: ...
    def reset_buffer(self) -> None: ...

def prepare_moe_ag_scatter_args(
    splits_gpu: torch.Tensor,
    scatter_index: torch.Tensor,
    ntokens: int,
    topk: int,
    num_weight_groups: int,
    ep_start: int,
    ep_experts: int,
    rank: int,
    world_size: int,
    tile_size_m: int,
    cp_stream: torch.cuda.Stream,
) -> Tuple[
    int,  # M_this_ep, int
    torch.Tensor,  # M_this_ep_pad, torch.Tensor(int32) on device
    torch.Tensor,  # gather_A_index
    torch.Tensor,  # scatter_D_index
    torch.Tensor,  # expert_index
    torch.Tensor,  # rank_start_index
    torch.Tensor,  # rank_end_index
]: ...

class AllGatherOp:
    def __init__(
        self,
        tp_group: dist.ProcessGroup,
        nnodes: int,
        max_m_dim: int,
        k_dim: int,
        input_dtype: torch.dtype,
    ) -> None: ...
    def run(
        self,
        input: torch.Tensor,
        input_scale: Optional[torch.Tensor],
        all_gather_option: AllGatherOption,
        cp_stream: ctypes.c_void_p,
    ) -> None: ...
    def local_barrier_buffer(self) -> torch.Tensor: ...
    def local_input_buffer(self) -> torch.Tensor: ...
    def local_input_scale_buffer(self) -> torch.Tensor: ...

def calc_scatter_index(
    choosed_experts: torch.Tensor,
    splits_gpu: torch.Tensor,
    num_expert: int,
) -> torch.Tensor: ...
