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

from typing import Optional, Sequence
import triton
import triton.language as tl
import torch
import flux


def _to_gemm_grouped_out_dtype(dtype: torch.dtype) -> torch.dtype:
    _DTYPE_TO_OUT_DTYPE = {
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.int8: torch.int32,
        torch.float8_e4m3fn: torch.bfloat16,
        torch.float8_e5m2: torch.bfloat16,
    }
    return _DTYPE_TO_OUT_DTYPE[dtype]


@triton.jit
def blocked_grouped_gemm_kernel(
    A,
    B,
    C,
    gather_a_index,
    scatter_d_index,
    expert_idx,
    M_pad,
    N,
    K,
    E,  # used to calculate the number of blocks
    M,
    A_stride_m,
    A_stride_k,
    B_stride_e,
    B_stride_k,
    B_stride_n,
    C_stride_m,
    C_stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M_pad, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    if pid >= num_block_m * num_block_n:
        return

    num_blocks_per_group = GROUP_SIZE_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(num_block_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gather_a = tl.load(gather_a_index + offs_token_id)
    offs_scatter_d = tl.load(scatter_d_index + offs_token_id)
    token_mask = offs_scatter_d < M

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offs_gather_a[:, None] * A_stride_m + offs_k[None, :] * A_stride_k

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_be = tl.load(expert_idx + pid_m)
    b_ptrs = B + offs_be * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n

    if A.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * A_stride_k
        b_ptrs += BLOCK_SIZE_K * B_stride_k

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if tl.constexpr(A.dtype.element_ty == tl.float8e4nv) or tl.constexpr(
        A.dtype.element_ty == tl.float8e5
    ):
        accumulator = accumulator.to(tl.bfloat16)
    else:
        accumulator = accumulator.to(A.dtype.element_ty)  # support BF16 now

    c_ptrs = C + offs_scatter_d[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _cdiv(a, b):
    return (a + b - 1) // b


class GemmGrouped(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input,
        weights,
        splits_cpu: Sequence[int],
        output: Optional[torch.Tensor] = None,
        BLOCK_SIZE_M: int = 32,
        BLOCK_SIZE_N: int = 32,
        BLOCK_SIZE_K: int = 32,
        GROUP_SIZE_M: int = 4,
        num_stages: int = 3,
        num_warps: int = 4,
    ):
        M, K = input.shape
        E, K, N = weights.shape

        # Allocates output.
        output_dtype = _to_gemm_grouped_out_dtype(input.dtype)
        if output is None:
            output = torch.empty((M, N), device=input.device, dtype=output_dtype)
        else:
            assert output.shape == (M, N), f"C.shape {output.shape}  vs ({M}, {N})"
            assert output.dtype == output_dtype, f"C.dtype {output.dtype} vs {output_dtype}"

        assert E == len(splits_cpu), f"E {E} vs len(splits_cpu) {len(splits_cpu)}"
        splits_cpu = torch.tensor(splits_cpu, device="cpu", dtype=torch.int32).pin_memory()
        splits_gpu = torch.empty(E, device=input.device, dtype=torch.int32)
        splits_gpu.copy_(splits_cpu, non_blocking=True)
        M_pad = int(((splits_cpu + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * BLOCK_SIZE_M).sum())
        num_blocks = M_pad // BLOCK_SIZE_M
        gather_a_index = torch.empty([M_pad], dtype=torch.int32, device=input.device)
        expert_idx = torch.empty([num_blocks], dtype=torch.int32, device=input.device)
        stream = torch.cuda.current_stream()
        flux.calc_moe_triton_blocked_gather_a(
            splits_gpu, 0, E, BLOCK_SIZE_M, gather_a_index, expert_idx, stream.cuda_stream
        )

        grid = lambda meta: (_cdiv(M_pad, meta["BLOCK_SIZE_M"]) * _cdiv(N, meta["BLOCK_SIZE_N"]),)
        blocked_grouped_gemm_kernel[grid](
            input,
            weights,
            output,
            gather_a_index,
            gather_a_index,
            expert_idx,
            M_pad,
            N,
            K,
            E,  # used to calculate the number of blocks
            M,
            input.stride(0),
            input.stride(1),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return output


def _matmul(A, B, out: Optional[torch.Tensor] = None):
    if A.dtype == torch.int8:
        return torch._int_mm(A, B, out=out)
    elif A.dtype == [torch.float8_e4m3fn, torch.float8_e5m2]:
        return torch.matmul(A.to(torch.bfloat16), B.to(torch.bfloat16), out=out)
    else:
        return torch.matmul(A, B, out=out)


def _torch_gemm_grouped_fp16(A, B, splits_cpu: Sequence[int]):
    M, K = A.shape
    E, K, N = B.shape
    assert len(splits_cpu) == E, f"len(splits_cpu) {len(splits_cpu)} vs {E}"
    output = torch.empty((M, N), device="cuda", dtype=_to_gemm_grouped_out_dtype(A.dtype))
    m_start = 0
    for e in range(E):
        m_end = m_start + splits_cpu[e]
        _matmul(A[m_start:m_end, :], B[e, ...], out=output[m_start:m_end, :])
        m_start = m_end
    return output


def _verify_gemm_grouped(M, N, K, E, dtype: torch.dtype, trans_a: bool, trans_b: bool):
    torch.manual_seed(0)
    if trans_a:
        A = torch.randn((K, M), device="cuda", dtype=dtype).T
    else:
        A = torch.randn((M, K), device="cuda", dtype=dtype)
    if trans_b:
        B = torch.randn((E, N, K), device="cuda", dtype=dtype).transpose(1, 2)
    else:
        B = torch.randn((E, K, N), device="cuda", dtype=dtype)

    print(f"B: {B.shape}")
    op = GemmGrouped()
    splits_cpu = torch.tensor([M // E for _ in range(E)], dtype=torch.int32, device="cuda")
    triton_output = op.forward(A, B, splits_cpu)
    torch_output = _torch_gemm_grouped_fp16(A, B, splits_cpu)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    rtol = 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


if __name__ == "__main__":
    _verify_gemm_grouped(512, 512, 512, 8, torch.float16, False, False)
    _verify_gemm_grouped(512, 512, 512, 8, torch.bfloat16, False, True)
