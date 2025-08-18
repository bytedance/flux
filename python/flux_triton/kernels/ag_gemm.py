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
import torch
import triton
import triton.language as tl
from flux_triton.extra import __syncthreads, ld_acquire, tid


def _is_A_gpu():
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (8, 0):
        return True
    return False


# key: (M, N, K, world_size)
# value: (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages)
TUNE_CONFIG_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_stages",
    "num_warps",
)
A800_TUNEMAP = {
    (1024, 6144, 12288, 8, torch.bfloat16): (128, 256, 32, 4, 3, 8),
    (2048, 6144, 12288, 8, torch.bfloat16): (256, 128, 32, 4, 3, 8),
    (4096, 6144, 12288, 8, torch.bfloat16): (128, 128, 32, 4, 4, 4),
    (8192, 6144, 12288, 8, torch.bfloat16): (256, 128, 32, 4, 3, 8),
    (1024, 6144, 12288, 8, torch.bfloat16): (128, 256, 32, 4, 3, 8),
    (2048, 6144, 12288, 8, torch.bfloat16): (256, 128, 32, 4, 3, 8),
    (4096, 6144, 12288, 8, torch.bfloat16): (128, 128, 32, 4, 4, 4),
    (8192, 6144, 12288, 8, torch.bfloat16): (256, 128, 32, 4, 3, 8),
}


def _as_triton_config_info(values):
    return {k: v for k, v in zip(TUNE_CONFIG_KEYS, values)}


def get_tune_config(
    M: int,  # TODO(houqi.1993) small M and large M should use different M
    N: int,
    K: int,
    world_size: int,
    trans_a: bool,
    trans_b: bool,
    dtype: torch.dtype,
):
    key = (M, N, K, world_size, dtype)
    if dtype in [torch.float16, torch.bfloat16]:
        if _is_A_gpu():
            if key in A800_TUNEMAP:
                return _as_triton_config_info(A800_TUNEMAP[key])
            return _as_triton_config_info((128, 256, 32, 2, 3, 8))
        return _as_triton_config_info((128, 128, 64, 4, 4, 4))
    if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        assert not trans_a and trans_b
        return _as_triton_config_info((64, 128, 32, 4, 4, 4))
    if dtype in [torch.int8]:
        assert not trans_a and trans_b
        return _as_triton_config_info((64, 128, 64, 4, 4, 4))
    assert False, f"not supported dtypes: {dtype}"


@triton.jit
def ag_gemm_kernel(
    # Pointers to matrices
    A,
    B,
    C,
    bias,
    input_scale,
    weight_scale,
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
    rank,
    world_size,
    barrier_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    BIAS_DIMS: tl.constexpr,
    INPUT_SCALE_DIMS: tl.constexpr,
    WEIGHT_SCALE_DIMS: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # threadblock swizzle
    #  no stream-k support. only split by m x n
    m_per_rank = M // world_size
    m_offset = m_per_rank * rank
    pid_m_offset = tl.cdiv(m_offset, BLOCK_SIZE_M)
    pid_m = (pid_m + pid_m_offset) % num_pid_m

    # wait for segment ready.
    # TODO(houqi.1993) don't use wait_eq_sys with for loop: there is a bug there and hangs. don't know why
    segment_start = pid_m * BLOCK_SIZE_M // m_per_rank
    segment_end = ((pid_m + 1) * BLOCK_SIZE_M - 1) // m_per_rank
    segment_end = min(segment_end, world_size - 1)
    segment = segment_start + tid(axis=0)
    if segment <= segment_end:
        while ld_acquire(barrier_ptr + segment, "sys") != 1:
            pass
    __syncthreads()

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if A.dtype.element_ty == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if A.dtype.element_ty == tl.int8:
        if INPUT_SCALE_DIMS == "m" and WEIGHT_SCALE_DIMS == "n":
            accumulator = accumulator.to(tl.float32)
            accumulator *= (
                tl.load(input_scale + offs_cm, mask=offs_cm < M)[:, None]
                * tl.load(weight_scale + offs_cn, mask=offs_cn < N)[None, :]
            )
            accumulator = accumulator.to(tl.bfloat16)
        else:
            tl.static_assert(False, "s8 gemm only support input_scale per-m and weight_scale per n")
    else:
        if IS_FP8:
            if INPUT_SCALE_DIMS == "0" and WEIGHT_SCALE_DIMS == "0":
                accumulator *= tl.load(input_scale) * tl.load(weight_scale)
            else:
                tl.static_assert(False, "fp8 only support input_scale/weight_scale dim = 0")

            accumulator = accumulator.to(tl.bfloat16)
        else:
            accumulator = accumulator.to(A.dtype.element_ty)

    if BIAS_DIMS == "n":
        accumulator += tl.load(bias + offs_cn, mask=offs_cn < N)[None, :]
    elif BIAS_DIMS == "mn":
        accumulator += tl.load(
            bias + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :], mask=c_mask
        )

    tl.store(c_ptrs, accumulator, mask=c_mask)
