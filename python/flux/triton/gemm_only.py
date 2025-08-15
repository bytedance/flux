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

from typing import List, Optional, Union

import torch
import triton
import triton.language as tl
from flux.testing import matmul_int8


@triton.jit
def persistent_matmul_kernel(
    # device tensor of matrices pointers
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  #
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_m_tiles * num_n_tiles

    while tile_idx < num_tiles:
        tile_m_idx = tile_idx // num_n_tiles
        tile_n_idx = tile_idx % num_n_tiles

        num_pid_in_group = GROUP_SIZE_M * num_n_tiles
        group_id = tile_idx // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
        tile_m_idx = first_pid_m + ((tile_idx % num_pid_in_group) % group_size_m)
        tile_n_idx = (tile_idx % num_pid_in_group) // group_size_m

        # do regular gemm here
        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A + offs_am[:, None] % M * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # hint to Triton compiler to do proper loop pipelining
            # tl.multiple_of(a_ptrs, [16, 16])
            # tl.multiple_of(b_ptrs, [16, 16])
            # assume full tile for now
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c = accumulator.to(tl.float16)

        offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn

        # assumes full tile for now
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

        # go to the next tile by advancing NUM_SM
        tile_idx += NUM_SM


def persistent_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (META["NUM_SM"],)
    persistent_matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        NUM_SM=92,
        num_warps=8,
        num_stages=3,
    )
    return [c]


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A,
    B,
    C,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  #
    GROUP_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_M, BLOCK_K] pointers
    # `b_ptrs` is a block of [BLOCK_K, BLOCK_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_M, BLOCK_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if A.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if A.dtype.element_ty == tl.float8e5:
        c = accumulator.to(tl.bfloat16)
    elif A.dtype.element_ty == tl.float8e4nv:
        c = accumulator.to(tl.bfloat16)
    elif A.dtype.element_ty == tl.float16:
        c = accumulator.to(A.dtype.element_ty)
    elif A.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(A.dtype.element_ty)
    else:
        c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(out_ptrs, c, mask=out_mask)


class Matmul(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        BLOCK_M: int = 32,
        BLOCK_N: int = 32,
        BLOCK_K: int = 32,
        GROUP_M: int = 4,
        num_stages: int = 3,
        num_warps: int = 8,
    ):
        # Check constraints.
        assert A.shape[1] == B.shape[0], f"Incompatible dimensions: {A.shape} vs {B.shape}"
        assert A.is_contiguous(), "Matrix A must be contiguous"
        M, K = A.shape
        K, N = B.shape
        # Allocates output.
        output_dtype = {
            torch.bfloat16: torch.bfloat16,
            torch.float16: torch.float16,
            torch.int8: torch.int32,
            torch.float8_e4m3fn: torch.bfloat16,
            torch.float8_e5m2: torch.bfloat16,
        }[A.dtype]
        if C is None:
            C = torch.empty((M, N), device=A.device, dtype=output_dtype)
        else:
            assert C.shape == (M, N), f"C.shape {C.shape}  vs ({M}, {N})"
            assert C.dtype == output_dtype, f"C.dtype {C.dtype} vs {output_dtype}"
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        matmul_kernel[grid](
            A,
            B,
            C,  #
            M,
            N,
            K,  #
            A.stride(0),
            A.stride(1),  #
            B.stride(0),
            B.stride(1),  #
            C.stride(0),
            C.stride(1),  #
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,  #
            GROUP_M=GROUP_M,  #
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return C


def _verify_matmul_fp16():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    op = Matmul()
    triton_output = op.forward(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    rtol = 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


def _verify_matmul_fp8():
    # NOTE: this fails on sm89 but not for sm90
    a = torch.empty((128, 8192), dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    a.fill_(1 / 32)
    b = torch.ones((8192, 8192), dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    b.fill_(3)
    op = Matmul()
    triton_output = op.forward(a, b)
    torch_output = torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    rtol = 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


def _verify_matmul_int8():
    # NOTE: this fails on sm89 but not for sm90
    a = torch.randint(-127, 127, (128, 8192), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 127, (8192, 8192), dtype=torch.int8, device="cuda")
    op = Matmul()
    triton_output = op.forward(a, b)
    torch_output = matmul_int8(a, b)
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    rtol = 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


if __name__ == "__main__":
    _verify_matmul_fp16()
    _verify_matmul_fp8()
    _verify_matmul_int8()
