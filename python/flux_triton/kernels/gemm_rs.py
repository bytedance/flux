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

from flux_triton.extra import (
    __ballot_sync,
    __syncthreads,
    arrive_inc,
    atomic_add,
    ffs,
    ld,
    ld_acquire,
    tid,
)

A800_TUNEMAP = {
    (1024, 12288, 6144, 8, torch.bfloat16): (128, 256, 32, 3, 8),
    (2048, 12288, 6144, 8, torch.bfloat16): (128, 256, 32, 3, 8),
    (4096, 12288, 6144, 8, torch.bfloat16): (128, 128, 32, 4, 4),
    (8192, 12288, 6144, 8, torch.bfloat16): (256, 128, 32, 3, 8),
    (1024, 12288, 6144, 8, torch.bfloat16): (128, 256, 32, 3, 8),
    (2048, 12288, 6144, 8, torch.bfloat16): (128, 256, 32, 3, 8),
    (4096, 12288, 6144, 8, torch.bfloat16): (128, 128, 32, 4, 4),
    (8192, 12288, 6144, 8, torch.bfloat16): (256, 128, 32, 3, 8),
}

TUNE_CONFIG_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "num_stages",
    "num_warps",
)


def _is_A_gpu():
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (8, 0):
        return True
    return False


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
            return _as_triton_config_info((128, 256, 32, 3, 8))
        return _as_triton_config_info((128, 128, 64, 4, 4))
    if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        assert not trans_a and trans_b
        return _as_triton_config_info((64, 128, 32, 4, 4))
    if dtype in [torch.int8]:
        assert not trans_a and trans_b
        return _as_triton_config_info((64, 128, 64, 4, 4))
    assert False, f"not supported dtypes: {dtype}"


@triton.jit
def add_continous_kernel(
    lhs,
    rhs,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    num_pid = tl.num_programs(axis=0)

    lhs_block_ptr = tl.make_block_ptr(
        base=lhs,
        shape=(N,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    rhs_block_ptr = tl.make_block_ptr(
        base=rhs,
        shape=(N,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out,
        shape=(N,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    for _ in range(pid, n_blocks, num_pid):
        tl.store(
            out_block_ptr,
            tl.load(lhs_block_ptr, boundary_check=(0,))
            + tl.load(rhs_block_ptr, boundary_check=(0,)),
            boundary_check=(0,),
        )
        lhs_block_ptr = tl.advance(lhs_block_ptr, [BLOCK_SIZE * num_pid])
        rhs_block_ptr = tl.advance(rhs_block_ptr, [BLOCK_SIZE * num_pid])
        out_block_ptr = tl.advance(out_block_ptr, [BLOCK_SIZE * num_pid])


@triton.jit
def gemm_rs_kernel(
    # Pointers to matrices
    A,  # [M, K]_Ti
    B,  # [K, N]_Ti
    bias,  # [1, N]_BF16 for INT8 or FP8, [M, N]_Ti for FP16/BF16
    C,  # [M, N]_To
    input_scale,  # [1]_FP32 for FP8, [M, 1]_FP32 for INT8
    weight_scale,  # [1]_FP32 for FP8, [1, N]_FP32 for INT8
    segments_info,
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
    gemm_tile_counter_ptr,
    gemm_ready_flag_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    BIAS_DIMS: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m_old = pid // num_pid_n
    pid_n_old = pid % num_pid_n

    thread_idx = tid(axis=0)
    laneid = thread_idx % 32
    segment_ptr = tl.where(laneid < world_size, segments_info + 4 * laneid, segments_info)
    size = tl.where(laneid < world_size, ld(segment_ptr + 1, scope="gpu"), -1)
    tile_m_start_origin = ld(segment_ptr + 2, scope="gpu")
    tile_m_start_new = ld(segment_ptr + 3, scope="gpu")

    index = __ballot_sync(
        0xFFFFFFFF,
        (size > 0 and pid_m_old >= tile_m_start_origin) and pid_m_old < tile_m_start_origin + size,
    )
    n = ffs(index) - 1

    segment_ptr = segments_info + 4 * n
    size = ld(segment_ptr + 1, scope="gpu")
    tile_m_start_origin = ld(segment_ptr + 2, scope="gpu")
    tile_m_start_new = ld(segment_ptr + 3, scope="gpu")
    tiled_m_offset = pid_m_old - tile_m_start_origin
    inner_idx = num_pid_n * tiled_m_offset + pid_n_old
    inner_m = inner_idx % size
    inner_n = inner_idx // size
    pid_m = inner_m + tile_m_start_new
    pid_n = inner_n
    m_per_rank = M // world_size

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
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if A.dtype.element_ty == tl.int8:
        c = (
            accumulator.to(tl.float32)
            * tl.load(input_scale + offs_cm, mask=offs_cm < M)[:, None]
            * tl.load(weight_scale + offs_cn, mask=offs_cn < N)[None, :]
        ).to(tl.bfloat16)
    elif A.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    elif A.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif A.dtype.element_ty == tl.float8e4nv:
        c = (accumulator * tl.load(input_scale) * tl.load(weight_scale)).to(tl.bfloat16)
    elif A.dtype.element_ty == tl.float8e5:
        c = (accumulator * tl.load(input_scale) * tl.load(weight_scale)).to(tl.bfloat16)
    else:
        tl.static_assert(False, "unsupported dtype")

    if BIAS_DIMS == "mn":
        c += tl.load(
            bias + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, mask=out_mask
        )
    elif BIAS_DIMS == "n":
        c += tl.load(bias + offs_cn * stride_cn, mask=(offs_cn < N))[None, :]
    tl.store(c_ptrs, c, mask=out_mask)

    # inc barrier
    segment_start = pid_m * BLOCK_SIZE_M // m_per_rank
    segment_end = ((pid_m + 1) * BLOCK_SIZE_M - 1) // m_per_rank
    segment_end = min(segment_end, world_size - 1)
    __syncthreads()
    segment = segment_start + tid(axis=0)
    if segment <= segment_end:
        m_start = m_per_rank * segment
        m_end = m_per_rank * (segment + 1) - 1
        tiled_m_start = m_start // BLOCK_SIZE_M
        tiled_m_end = m_end // BLOCK_SIZE_M
        tiled_m_size = tiled_m_end - tiled_m_start + 1
        tiled_n = tl.cdiv(N, BLOCK_SIZE_N)
        if (
            atomic_add(gemm_tile_counter_ptr + segment, 1, semantic="release", scope="gpu")
            == tiled_n * tiled_m_size - 1
        ):
            atomic_add(gemm_ready_flag_ptr + segment, 1, semantic="relaxed", scope="gpu")


@triton.jit
def gemm_rs_nvlink_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptrs,
    # Matrix dimensions
    M,
    N,
    K_per_rank,
    blocks_m_per_rank,
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
    barrier_ptrs,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
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
    # k_per_rank = K // world_size
    m_per_rank = M // world_size
    m_offset = m_per_rank * rank
    pid_m_offset = tl.cdiv(m_offset, BLOCK_SIZE_M)
    pid_m = (pid_m + pid_m_offset) % num_pid_m

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
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_per_rank, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K_per_rank - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K_per_rank - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    # blocks_m_per_rank = tl.cdiv(num_pid_m, world_size)
    target_rank_id = pid_m // blocks_m_per_rank
    target_pid_m = (rank * blocks_m_per_rank + pid_m % blocks_m_per_rank) % num_pid_m

    c_ptr = tl.load(c_ptrs + target_rank_id).to(tl.pointer_type(tl.float16))
    c_tile_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(target_pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_tile_ptr, c, boundary_check=(0, 1))

    # tile done, increase barrier
    barrier_ptr = tl.load(barrier_ptrs + target_rank_id).to(tl.pointer_type(tl.int32))
    arrive_inc(barrier_ptr + rank, tid(axis=0), 1, "sys")


@triton.jit
def reduce_nvlink_kernel(
    c_ptr,
    out_ptr,
    # shape of matrix
    M,
    N,
    # strides
    stride_m,
    stride_n,
    # distributed arguments
    rank,
    world_size,
    barrier_ptr,
    # reduce tile shape
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # gemm tile shape
    GEMM_BLOCK_M: tl.constexpr,
    GEMM_BLOCK_N: tl.constexpr,
):
    m_per_rank = tl.cdiv(M, world_size)
    gemm_m_blocks_per_rank = tl.cdiv(m_per_rank, GEMM_BLOCK_M)
    gemm_n_blocks = tl.cdiv(N, GEMM_BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    pid = tl.program_id(axis=0)
    reduce_m_blocks_per_rank = tl.cdiv(m_per_rank, BLOCK_M)
    reduce_n_blocks_per_rank = tl.cdiv(N, BLOCK_N)
    pid_m = pid // reduce_n_blocks_per_rank
    pid_n = pid % reduce_n_blocks_per_rank

    for rid in range(0, world_size):
        swizzle_rid = (rid + rank) % world_size
        if tid(axis=0) == 0:
            while (
                ld_acquire(barrier_ptr + swizzle_rid, "sys")
                != gemm_m_blocks_per_rank * gemm_n_blocks
            ):
                pass
        __syncthreads()
        full_offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M) + swizzle_rid * m_per_rank) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        ptrs = c_ptr + (full_offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
        data = tl.load(ptrs)
        acc += data.to(tl.float32)

    res = acc.to(tl.float16)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    ptrs = out_ptr + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
    tl.store(ptrs, res)
