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

from typing import Optional

import torch
import torch.distributed
import triton
from flux_triton.kernels.ag_gemm import ag_gemm_kernel, get_tune_config

import flux
from flux.util import is_fp8_dtype


class AgGemmTriton(torch.nn.Module):
    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        input_dtype: torch.dtype,
        max_m: int = 8192,
        k: int = 8192,
        transpose_weight: bool = True,
    ):
        self.pg = pg
        self.rank: int = pg.rank()
        self.world_size: int = pg.size()
        self.max_m: int = max_m
        assert max_m % self.world_size == 0, f"{max_m} % {self.world_size} != 0"
        self.max_m_per_rank: int = max_m // self.world_size
        self.input_dtype: torch.dtype = input_dtype
        TYPE_DICT = {
            torch.float16: torch.float16,
            torch.bfloat16: torch.bfloat16,
            torch.float8_e4m3fn: torch.bfloat16,
            torch.float8_e5m2: torch.bfloat16,
            torch.int8: torch.bfloat16,
        }
        self.output_dtype: torch.dtype = TYPE_DICT[input_dtype]
        self.cp_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)
        self.transpose_weight = transpose_weight
        self.ag_op = flux.AllGatherOp(
            self.pg, 1, max_m, k, input_dtype  # TODO(houqi.1993) does not support multi nodes
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,  # TODO(houqi.1993) not defined yet
        gathered_input: Optional[torch.Tensor] = None,
        ag_option: flux.AllGatherOption = flux.AllGatherOption(),
        fast_accum: bool = False,  # TODO(houqi.1993) seems no fast_accum for triton?
    ):
        stream = torch.cuda.current_stream()
        is_s8 = self.input_dtype == torch.int8
        is_fp8 = is_fp8_dtype(self.input_dtype)
        # Check constraints.
        if self.transpose_weight:
            weight = weight.t()
        assert (
            x.shape[1] == weight.shape[0]
        ), f"Incompatible dimensions: {x.shape} vs {weight.shape}"
        assert x.is_contiguous(), "Matrix A must be contiguous"
        M_per_rank, K = x.shape
        M = M_per_rank * self.world_size
        assert self.max_m >= M, f"{M_per_rank} > {self.max_m_per_rank}"
        K, N = weight.shape
        # Allocates output.
        c = torch.empty((M, N), device=weight.device, dtype=self.output_dtype)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

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
            x, input_scale if do_gather_input_scale else None, ag_option, self.cp_stream.cuda_stream
        )

        ag_gemm_kernel[grid](
            input_tensor,
            weight,
            c,  #
            bias,
            input_scale_tensor,
            weight_scale,
            M_per_rank * self.world_size,
            N,
            K,  #
            input_tensor.stride(0),
            input_tensor.stride(1),  #
            weight.stride(0),
            weight.stride(1),  #
            c.stride(0),
            c.stride(1),  #
            self.rank,
            self.world_size,
            barrier,
            BIAS_DIMS="" if bias is None else ("n" if is_s8 or is_fp8 else "mn"),
            INPUT_SCALE_DIMS=("m" if is_s8 else (None if not is_fp8 else "0")),
            WEIGHT_SCALE_DIMS=("n" if is_s8 else (None if not is_fp8 else "0")),
            IS_FP8=is_fp8,
            **get_tune_config(
                M_per_rank * self.world_size, N, K, self.world_size, False, True, self.input_dtype
            ),
        )
        if gathered_input is not None:
            gathered_input.copy_(input_tensor)

        return c, gathered_input
