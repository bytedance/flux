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

import flux
from flux_triton.kernels.moe_gather_rs import (
    get_tune_config,
    run_moe_gather_rs_grouped_gemm,
    run_moe_gather_rs_grouped_gemm_with_groups,
)


class MoeGatherRsOp(torch.nn.Module):

    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        max_ntokens: int,
        hidden: int,
        E_total: int,
        topk: int,
        ep_size: int,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        N_SPLIT: int = 4,
        do_all_reduce: bool = False,
        use_read_mode: bool = False,
    ):
        super(MoeGatherRsOp, self).__init__()
        self.tp_group = tp_group
        self.rank = tp_group.rank()
        self.world_size = tp_group.size()
        self.max_ntokens = max_ntokens
        self.N = hidden
        self.E = E_total
        self.ep_size = ep_size
        self.tp_size = self.world_size // self.ep_size
        self.tp_rank = self.rank % self.tp_size
        self.ep_rank = self.rank // self.tp_size
        self.E_this_ep = E_total // self.ep_size
        self.ep_start = self.ep_rank * self.E_this_ep
        self.ep_end = self.ep_start + self.E_this_ep
        self.topk = topk
        self.max_M = max_ntokens * self.topk
        self.input_dtype = input_dtype
        self.output_dtype = input_dtype if output_dtype is None else output_dtype
        assert input_dtype in [torch.float16, torch.bfloat16, torch.int8], input_dtype
        self.N_SPLIT = N_SPLIT
        if (self.N / N_SPLIT) % 1024 != 0:
            assert self.N % 1024 == 0, f"self.N({self.N}) must be divisible by 1024"
            self.N_SPLIT = self.N // 1024
            print(f"Warning: (n/split_n) % 1024 != 0, set N_SPLIT to {self.N_SPLIT}")

        self.gemm_ready_flags = flux.create_tensor_list(
            (2 * self.N_SPLIT,), dtype=torch.int32, pg=self.tp_group, ring_mode=True
        )
        self.gemm_ready_flag = self.gemm_ready_flags[self.rank]
        self.cp_stream = torch.cuda.Stream(priority=-1)
        self.topk_reduce_scatter_op = flux.TopkReduceScatterOp(
            tp_group,
            self.max_ntokens,
            self.N,
            self.topk,
            self.output_dtype,
            self.E,
            self.ep_size,
            self.gemm_ready_flags,
            self.N_SPLIT,
            do_all_reduce,
            use_read_mode=use_read_mode,
        )
        self.group_barrier = flux.GroupBarrier(self.tp_group, False)

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        splits_cpu: torch.Tensor,
        scatter_index: torch.Tensor,
        output: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_vec_scale: torch.Tensor,
        fast_accum: bool = True,  # triton current only supports FP8 FastAcc
        transpose_weight: bool = True,
    ):
        M_this_ep, K = input.shape
        if self.ep_size == 1:
            assert M_this_ep % self.topk == 0
        assert M_this_ep <= self.max_M, f"{M_this_ep} vs {self.max_M}"
        ep_nexperts, N = self.E_this_ep, self.N
        if transpose_weight:
            weight = weight.transpose(1, 2)
        assert weight.shape == (ep_nexperts, K, N), f"{weight.shape} vs {(ep_nexperts, K, N)}"

        ep_start, ep_end = self.ep_start, self.ep_end

        config = get_tune_config(
            M_this_ep,
            N,
            K,
            self.world_size,
            False,
            transpose_weight,
            input.dtype,
        )
        BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
        m_tiles = int(((splits_cpu[ep_start:ep_end] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M).sum())
        expert_idx = torch.empty([m_tiles], dtype=torch.int32, device="cuda")
        gather_a_index = torch.empty([m_tiles * BLOCK_SIZE_M], dtype=torch.int32, device="cuda")

        self.gemm_ready_flag.zero_()
        self.topk_reduce_scatter_op.reset_buffer()

        stream = torch.cuda.current_stream()
        self.group_barrier.barrier_all(stream.cuda_stream)

        splits_gpu = torch.empty_like(splits_cpu, device="cuda")
        splits_gpu.copy_(splits_cpu, non_blocking=False)
        flux.calc_moe_triton_blocked_gather_a(
            splits_gpu,
            ep_start,
            self.E_this_ep,
            BLOCK_SIZE_M,
            gather_a_index,
            expert_idx,
            stream.cuda_stream,
        )

        gemm_out = torch.empty((M_this_ep, N), dtype=self.output_dtype, device="cuda")
        self.cp_stream.wait_stream(stream)
        if M_this_ep == 0:
            self.gemm_ready_flag.fill_(1)
        else:
            run_moe_gather_rs_grouped_gemm(
                input,
                weight,
                gemm_out,
                input_scale,
                weight_scale,
                output_vec_scale,
                gather_a_index,
                expert_idx,
                gather_a_index.shape[0],
                N,
                K,
                ep_nexperts,
                M_this_ep,
                self.gemm_ready_flag,
                N_SPLIT=self.N_SPLIT,
                config=config,
            )

        output = self.topk_reduce_scatter_op.run(
            [gemm_out],
            output,
            self.ep_start,
            self.E_this_ep,
            splits_gpu,
            scatter_index,
            None,  # output_vec_scales. done in matmul stage
            3,  # num_thread_blocks
            self.cp_stream.cuda_stream,
        )
        torch.cuda.current_stream().wait_stream(self.cp_stream)
        return output

    def forward_multiple(
        self,
        input_list: List[torch.Tensor],
        weight_list: List[torch.Tensor],
        splits_cpu: torch.Tensor,
        scatter_index: torch.Tensor,
        output: Optional[torch.Tensor],
        input_scale_list: List[torch.Tensor],
        weight_scale_list: List[torch.Tensor],
        output_vec_scale_list: List[torch.Tensor],
        fast_accum: bool = True,
        transpose_weight: bool = True,
    ):
        L = len(input_list)
        assert L >= 1 and L <= 2, L
        assert len(weight_list) == L, f"{len(weight_list)} vs {L}"
        assert len(input_scale_list) == L, f"{len(input_scale_list)} vs {L}"
        assert len(weight_scale_list) == L, f"{len(weight_scale_list)} vs {L}"
        assert len(output_vec_scale_list) == L, f"{len(output_vec_scale_list)} vs {L}"

        M_this_ep, K = input_list[0].shape
        if self.ep_size == 1:
            assert M_this_ep % self.topk == 0
        assert M_this_ep <= self.max_M, f"{M_this_ep} vs {self.max_M}"
        ep_nexperts, N = self.E_this_ep, self.N

        if transpose_weight:
            weight_list = [w.transpose(1, 2) for w in weight_list]
        for w in weight_list:
            assert len(w.shape) == 3, w.shape
            assert w.shape == (ep_nexperts, K, N), f"{w.shape} vs {(ep_nexperts, K, N)}"

        ep_start, ep_end = self.ep_start, self.ep_end

        config = get_tune_config(
            M_this_ep,
            N,
            K,
            self.world_size,
            False,
            transpose_weight,
            input_list[0].dtype,
        )
        BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
        m_tiles = int(((splits_cpu[ep_start:ep_end] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M).sum())
        expert_idx = torch.empty([m_tiles], dtype=torch.int32, device="cuda")
        gather_index_pad_gpu = torch.empty(
            [m_tiles * BLOCK_SIZE_M], dtype=torch.int32, device="cuda"
        )

        stream = torch.cuda.current_stream()
        self.topk_reduce_scatter_op.reset_buffer()
        self.gemm_ready_flag.zero_()
        self.group_barrier.barrier_all(stream.cuda_stream)

        splits_gpu = torch.empty_like(splits_cpu, device="cuda")
        splits_gpu.copy_(splits_cpu, non_blocking=False)
        flux.calc_moe_triton_blocked_gather_a(
            splits_gpu,
            ep_start,
            self.E_this_ep,
            BLOCK_SIZE_M,
            gather_index_pad_gpu,
            expert_idx,
            stream.cuda_stream,
        )

        gemm_outs = [
            torch.empty((M_this_ep, N), dtype=self.output_dtype, device="cuda") for _ in range(L)
        ]
        self.cp_stream.wait_stream(stream)
        if M_this_ep == 0:
            self.gemm_ready_flag.fill_(1)
        else:
            run_moe_gather_rs_grouped_gemm_with_groups(
                input_list,
                weight_list,
                gemm_outs,
                input_scale_list,
                weight_scale_list,
                output_vec_scale_list,
                gather_index_pad_gpu,
                expert_idx,
                gather_index_pad_gpu.shape[0],
                N,
                K,
                ep_nexperts,
                M_this_ep,
                self.gemm_ready_flag,
                self.N_SPLIT,
                config,
            )

        output = self.topk_reduce_scatter_op.run(
            gemm_outs,
            output,
            self.ep_start,
            self.E_this_ep,
            splits_gpu,
            scatter_index,
            None,  # output vec scales
            3,  # TODO(houqi.1993)
            self.cp_stream.cuda_stream,
        )

        torch.cuda.current_stream().wait_stream(self.cp_stream)
        return output
