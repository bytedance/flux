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


from typing import List, Optional

import torch
import torch.distributed
import triton

import flux
from flux_triton.kernels.moe_ag_scatter import run_moe_ag_scatter_grouped_gemm, get_triton_algo_info


def cdiv(a: int, b: int):
    return (a + b - 1) // b


class MoeAgScatterOp(torch.nn.Module):
    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        ep_group: torch.distributed.ProcessGroup,
        max_ntokens: int,
        hidden: int,
        ffn_hidden: int,
        nexperts: int,
        topk: int,
        input_dtype: torch.dtype,
    ):
        self.tp_group = tp_group
        self.ep_group = ep_group
        self.ffn_tp_size = tp_group.size() // ep_group.size()
        self.ffn_size_shard = ffn_hidden // self.ffn_tp_size
        self.rank = tp_group.rank()
        self.world_size = tp_group.size()
        self.ep_rank = ep_group.rank()
        self.max_ntokens = max_ntokens
        self.hidden = hidden
        self.ffn_hidden = ffn_hidden
        self.nexperts = nexperts
        self.topk = topk
        self.input_dtype = input_dtype
        assert input_dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.int8,
        ], input_dtype

        self.ep_nexperts = nexperts // ep_group.size()
        self.ep_start = self.ep_rank * self.ep_nexperts
        self.cp_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)
        self.ag_op = flux.AllGatherOp(
            self.tp_group,
            1,
            max_ntokens,
            self.hidden,
            input_dtype,  # TODO(houqi.1993) does not support multi nodes
        )

    def forward(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,  # shape [E, N, K]
        bias: Optional[torch.Tensor],  # shape [E, 1, K]
        input_scale: Optional[torch.Tensor],  # TODO(houqi.1993) support no groups
        weight_scale: Optional[torch.Tensor],  # TODO(houqi.1993) support no groups
        output_scale: torch.Tensor,
        splits_gpu: torch.Tensor,
        scatter_index: torch.Tensor,
        outputs_buf: torch.Tensor,
        fast_accum: bool = True,
        transpose_weight: bool = True,
        ag_option: flux.AllGatherOption = flux.AllGatherOption(),
        gathered_input: torch.Tensor = None,
    ):
        """
        input: [ntokens, hidden]
        weights: [E, N, K]
        """
        ntokens, topk = scatter_index.shape
        ntokens_this_rank, K = input.shape
        assert (
            ntokens == ntokens_this_rank * self.world_size
        ), f"{ntokens} vs {ntokens_this_rank * self.world_size}"
        assert ntokens <= self.max_ntokens, f"{ntokens} vs {self.max_ntokens}"
        assert self.topk == topk, f"{self.topk} vs {topk}"
        # TODO(houqi.1993) should here assert K and self.hidden
        stream = torch.cuda.current_stream()
        is_s8 = self.input_dtype == torch.int8
        is_fp8 = flux.util.is_fp8_dtype(input.dtype)
        assert not (
            is_fp8 and int(triton.__version__.split(".")[0]) < 3
        ), f"triton {triton.__version__} not support FP8 dtype"

        M = ntokens
        do_gather_input_scale = is_s8 and input_scale is not None
        input_tensor = self.ag_op.local_input_buffer()[0:M, :]
        input_scale_tensor = (
            self.ag_op.local_input_scale_buffer()[0:M, :] if do_gather_input_scale else input_scale
        )
        barrier = self.ag_op.local_barrier_buffer()
        self.cp_stream.wait_stream(stream)
        # materialize ag_option
        if ag_option.use_cuda_core_local is None:
            ag_option.use_cuda_core_local = do_gather_input_scale
        if ag_option.use_cuda_core_ag is None:
            ag_option.use_cuda_core_ag = do_gather_input_scale
        if ag_option.fuse_sync is None:
            ag_option.fuse_sync = ag_option.use_cuda_core_local
        if ag_option.use_read is None:
            ag_option.use_read = False  # push is better
        self.ag_op.run(
            input,
            input_scale if do_gather_input_scale else None,
            ag_option,
            self.cp_stream.cuda_stream,
        )

        if transpose_weight:
            E, N, K_ = weights.shape
        else:
            E, K_, N = weights.shape
        assert K == K_, f"{K} vs {K_}"

        assert splits_gpu.is_cuda

        algo_info = get_triton_algo_info(input.dtype, with_bias=bias is not None)

        (
            M_this_ep,
            M_this_ep_pad,  # Tensor
            gather_A_index,
            scatter_D_index,
            expert_idx,
            rank_start_idx,
            rank_end_idx,
        ) = flux.prepare_moe_ag_scatter_args(
            splits_gpu,
            scatter_index,
            ntokens,
            topk,
            1,  # TODO(houqi.1993) group inputs
            self.ep_start,
            self.ep_nexperts,
            self.rank,
            self.world_size,
            algo_info["BLOCK_SIZE_M"],
            stream.cuda_stream,
        )
        # stream.synchronize()
        # print(f"M_this_ep: {M_this_ep}", flush=True)
        # rank = self.rank
        # torch.save(gather_A_index, f"gather_A_index_{rank}.pt")
        # torch.save(scatter_D_index, f"scatter_D_index_{rank}.pt")
        # torch.save(expert_idx, f"expert_idx_{rank}.pt")
        # torch.save(rank_start_idx, f"rank_start_idx_{rank}.pt")
        # torch.save(rank_end_idx, f"rank_end_idx_{rank}.pt")
        # torch.cuda.synchronize()
        output_dtype = input.dtype if input.dtype != torch.int8 else torch.bfloat16
        output = (
            torch.empty((M_this_ep, N), dtype=output_dtype, device="cuda")
            if outputs_buf is None
            else outputs_buf
        )

        if transpose_weight:
            weights = weights.transpose(1, 2)
        output = run_moe_ag_scatter_grouped_gemm(
            input_tensor,  # use input_tensor ptr anyway. use no shape
            weights,
            output,
            bias,
            input_scale_tensor,
            weight_scale,
            output_scale,
            gather_A_index,
            scatter_D_index,
            expert_idx,
            rank_start_idx,
            rank_end_idx,
            M_this_ep_pad,
            N,
            K,
            E,
            M_this_ep,
            barrier,
            **algo_info,
        )
        if gathered_input is not None:
            gathered_input.copy_(input_tensor)

        return output
