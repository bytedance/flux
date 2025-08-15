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
from flux_triton.kernels.gemm_rs import (
    add_continous_kernel,
    gemm_rs_kernel,
    gemm_rs_nvlink_kernel,
    get_tune_config,
    reduce_nvlink_kernel,
)

import flux
from cuda import cuda, cudart


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


class GemmRSTritonPCIe(torch.nn.Module):
    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        max_m: int = 8192,
        n: int = 8192,
    ):
        self.pg = pg
        self.rank: int = pg.rank()
        self.world_size: int = pg.size()
        self.max_m: int = max_m
        assert max_m % self.world_size == 0, f"max_m({max_m}) % world_size({self.world_size}) != 0"
        self.n: int = n
        self.input_dtype: torch.dtype = input_dtype
        self.output_dtype: torch.dtype = output_dtype
        self.barriers: List[torch.Tensor] = flux.create_tensor_list(
            (
                3,
                self.world_size,
            ),  # 1st for ready flag, 2nd for ready tile counter, 3rd for reduce flag
            torch.int32,
            self.pg,
        )
        self.barrier = self.barriers[self.rank]
        self.gemm_ready_flags = [barrier[0, :] for barrier in self.barriers]
        self.gemm_tile_counters = [barrier[1, :] for barrier in self.barriers]
        self.reduce_ready_flags = [barrier[2, :] for barrier in self.barriers]
        self.gemm_ready_flag = self.gemm_ready_flags[self.rank]
        self.gemm_tile_counter = self.gemm_tile_counters[self.rank]
        self.reduce_ready_flag = self.reduce_ready_flags[self.rank]

        self.cp_stream: torch.cuda.Stream = torch.cuda.Stream(
            priority=-1
        )  # high priority stream is needed
        self.output_tensors: List[torch.Tensor] = flux.create_tensor_list(
            (max_m, n), self.output_dtype, self.pg
        )
        assert len(self.output_tensors) == self.world_size
        self.output_tensor: torch.Tensor = self.output_tensors[self.rank]
        self.reduce_tensors: list[torch.Tensor] = flux.create_tensor_list(
            (max_m, n), self.output_dtype, self.pg
        )
        assert len(self.reduce_tensors) == self.world_size
        self.reduce_tensor: torch.Tensor = self.reduce_tensors[self.rank]
        self.group_barrier = flux.GroupBarrier(self.pg, False)

    def set_copy_done(self, rank: int, segment: int, stream: torch.cuda.Stream):
        (err,) = cuda.cuStreamWriteValue32(
            stream.cuda_stream,
            self.reduce_ready_flags[rank][segment].data_ptr(),
            1,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        CUDA_CHECK(err)

    def wait_gemm_ready(self, rank: int, segment: int, stream: torch.cuda.Stream):
        (err,) = cuda.cuStreamWaitValue32(
            stream.cuda_stream,
            self.gemm_ready_flags[rank][segment].data_ptr(),
            1,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)

    def wait_copy_ready(self, rank: int, segment: int, stream: torch.cuda.Stream):
        (err,) = cuda.cuStreamWaitValue32(
            stream.cuda_stream,
            self.reduce_ready_flags[rank][segment].data_ptr(),
            1,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)

    def run_reduce_scatter_ring_push_1d(
        self,
        M: int,
        output: Optional[torch.Tensor],
        stream: torch.cuda.Stream,
    ):
        M_per_rank = M // self.world_size
        if output:
            assert output.dtype == self.output_dtype
            M_, N_ = output.shape
            assert M_ == M_per_rank and N_ == self.n and output.is_cuda

        # M_per_rank = M // self.world_size
        to_rank = (self.rank - 1 + self.world_size) % self.world_size
        with torch.cuda.stream(stream):
            for stage in range(self.world_size):
                send_segment = (self.rank + stage + 1) % self.world_size
                dst = self.reduce_tensors[to_rank][
                    send_segment * M_per_rank : (send_segment + 1) * M_per_rank, :
                ]
                src = (
                    output
                    if stage == self.world_size - 1 and output is not None
                    else self.output_tensor[
                        send_segment * M_per_rank : (send_segment + 1) * M_per_rank, :
                    ]
                )
                self.wait_gemm_ready(self.rank, send_segment, stream)
                if stage != 0:
                    self.wait_copy_ready(self.rank, send_segment, stream)
                    buffer = self.reduce_tensors[self.rank][
                        send_segment * M_per_rank : (send_segment + 1) * M_per_rank, :
                    ]
                    # src.add_(buffer)
                    add_continous_kernel[(16,)](
                        src,
                        buffer,
                        src,
                        dst.numel(),
                        num_warps=32,
                        BLOCK_SIZE=1024 * 8 * 4,
                    )
                if stage == self.world_size - 1:
                    break
                dst.copy_(src)
                self.set_copy_done(to_rank, send_segment, stream)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,  # TODO(wenlei.bao) how to handle output_scale?
        transpose_weight: bool = False,
        fast_accum: bool = False,  # TODO(houqi.1993) how to do triton fast_acc?
    ):
        if not transpose_weight:
            weight = weight.t()

        stream = torch.cuda.current_stream()
        # Check constraints.
        M, K = x.shape
        M_per_rank = M // self.world_size
        assert K == weight.shape[0], f"Incompatible dimensions: {x.shape} vs {weight.shape}"
        assert x.is_contiguous(), "Matrix A must be contiguous"
        assert M % self.world_size == 0, f"{M} % {self.world_size} != 0"
        assert self.max_m >= M, f"{M} > {self.max_m}"
        K, N = weight.shape
        assert self.n == N, f"{self.n} != {N}"
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        self.barrier.zero_()
        self.group_barrier.barrier_all(stream.cuda_stream)

        self.cp_stream.wait_stream(stream)
        segments_info_cpu = torch.zeros(
            (self.world_size * 4,), dtype=torch.int16, device="cpu", pin_memory=True
        )

        config = get_tune_config(
            M, N, K, self.world_size, False, not transpose_weight, self.input_dtype
        )
        TILE_SIZE_M = config["BLOCK_SIZE_M"]
        TILE_SIZE_N = config["BLOCK_SIZE_N"]

        flux.calc_gemm_rs_threadblock_segments_info(
            segments_info_cpu,
            (M, N),
            (TILE_SIZE_M, TILE_SIZE_N),
            self.rank,
            self.world_size,
            self.world_size,
            1,  # nnodes. only nodes==1 supported
            False,  # use 2d ring. not implemented here
            False,  # per-tile flag. not implemented here
        )
        segments_info = torch.zeros_like(segments_info_cpu, device="cuda")
        segments_info.copy_(segments_info_cpu, non_blocking=True)
        is_s8 = self.input_dtype == torch.int8
        is_fp8 = flux.util.is_fp8_dtype(x.dtype)
        # NOTE: strange but necessary. to align with flux. better put it outside
        bias = None if is_s8 and self.rank != 0 else bias
        # run gemm
        gemm_rs_kernel[grid](
            x,
            weight,
            bias,
            self.output_tensor,  #
            input_scale,
            weight_scale,
            segments_info,
            M,
            N,
            K,  #
            x.stride(0),
            x.stride(1),  #
            weight.stride(0),
            weight.stride(1),  #
            self.output_tensor.stride(0),
            self.output_tensor.stride(1),  #
            self.rank,
            self.world_size,
            self.gemm_tile_counter,
            self.gemm_ready_flag,
            BIAS_DIMS=("none" if bias is None else ("n" if is_s8 or is_fp8 else "mn")),
            **config,
        )
        self.run_reduce_scatter_ring_push_1d(M, output, self.cp_stream)
        stream.wait_stream(self.cp_stream)
        # print(self.barrier)
        self.group_barrier.barrier_all(stream.cuda_stream)
        return self.output_tensor[self.rank * M_per_rank : (self.rank + 1) * M_per_rank, :]


class GemmRSTritonNVLink(torch.nn.Module):
    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        weight: torch.Tensor,
        max_m: int,
        n: int,
        k: int,
        fused: bool = False,
        dump_ptx: bool = False,
    ):
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()
        self.weight = weight
        self.max_m = max_m
        self.n = n
        assert max_m % self.world_size == 0, f"max_m={max_m} % world_size={self.world_size} != 0"
        assert (
            weight.shape[0] * self.world_size == k
        ), f"K={k} != weight.shape[1]={weight.shape[0]}*world_size={self.world_size}"
        assert weight.shape[1] == n
        self.max_m_per_rank = max_m // self.world_size
        self.dtype = weight.dtype
        self.output_dtype = weight.dtype
        self.barriers: List[torch.Tensor] = flux.create_tensor_list(
            (self.world_size, 1), torch.int32, self.pg
        )
        self.barrier_ptrs = torch.tensor([ptr.data_ptr() for ptr in self.barriers]).cuda()
        self.barrier = self.barriers[self.rank]
        self.rs_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.output_tensors: List[torch.Tensor] = flux.create_tensor_list(
            (self.max_m, n), self.output_dtype, self.pg
        )
        self.output_tensor_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in self.output_tensors]
        ).cuda()
        assert len(self.output_tensors) == self.world_size
        self.output_tensor: torch.Tensor = self.output_tensors[self.rank]
        self.output = torch.zeros(
            [self.max_m_per_rank, self.n], dtype=self.output_tensor.dtype
        ).cuda()
        # fusion control
        self.fused = fused
        # debug control
        self.dump_ptx = dump_ptx
        self.group_barrier = flux.GroupBarrier(self.pg, False)

    def forward(self, x: torch.Tensor):
        stream = torch.cuda.current_stream()
        assert x.shape[1] == self.weight.shape[0]
        assert x.is_contiguous(), "Matrix A must be contiguous"
        M, K_per_rank = x.shape
        K = K_per_rank * self.world_size
        _, N = self.weight.shape
        assert self.max_m >= M, "max_m < M"

        self.barrier.zero_()
        self.group_barrier.barrier_all(stream.cuda_stream)

        def ceil(a, b):
            return (a + b - 1) // b

        m_per_rank = ceil(M, self.world_size)
        blocks_m_per_rank = ceil(m_per_rank, 128)
        gemm_blocks = ceil(M, 128) * ceil(N, 128)
        reduce_blocks = ceil(m_per_rank, 128) * ceil(N, 128)

        if self.fused:
            raise NotImplementedError()

        else:
            # GEMM RS hyper parameters
            GEMM_BLOCK_M = 128
            GEMM_BLOCK_N = 128
            GEMM_BLOCK_K = 32
            GEMM_GROUP_M_SIZE = 8
            REDUCE_BLOCK_M = 128
            REDUCE_BLOCK_N = 128
            with torch.cuda.stream(stream):
                grid = lambda META: (
                    triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
                )
                gemm_rs_nvlink_kernel[grid](
                    x,
                    self.weight,
                    self.output_tensor_ptrs,
                    M,
                    N,
                    K_per_rank,
                    blocks_m_per_rank,
                    x.stride(0),
                    x.stride(1),
                    self.weight.stride(0),
                    self.weight.stride(1),
                    self.output_tensor.stride(0),
                    self.output_tensor.stride(1),
                    self.rank,
                    self.world_size,
                    self.barrier_ptrs,
                    GEMM_BLOCK_M,
                    GEMM_BLOCK_N,
                    GEMM_BLOCK_K,
                    GEMM_GROUP_M_SIZE,
                    num_stages=4,
                    num_warps=4,
                )

            with torch.cuda.stream(self.rs_stream):
                grid = lambda META: (
                    triton.cdiv(m_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
                )
                reduce_nvlink_kernel[grid](
                    self.output_tensor,
                    self.output,
                    M,
                    N,
                    self.output.stride(0),
                    self.output.stride(1),
                    self.rank,
                    self.world_size,
                    self.barrier,
                    REDUCE_BLOCK_M,
                    REDUCE_BLOCK_N,
                    GEMM_BLOCK_M,
                    GEMM_BLOCK_N,
                )
            stream.wait_stream(self.rs_stream)
        if self.dump_ptx:
            if self.rank == 0:
                if self.fused:
                    raise NotImplementedError()
                else:
                    with open("debug_output_GEMM_RS_NVLink_Triton.ptx", "w") as a:
                        print(list(gemm_rs_nvlink_kernel.cache[0].values())[0].asm["ptx"], file=a)
        self.group_barrier.barrier_all(stream.cuda_stream)
        return self.output[:m_per_rank, :]
