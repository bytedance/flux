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

from flux.triton.extra import (
    __ballot_sync,
    __shfl_sync_i32,
    __shfl_up_sync_i32,
    atomic_add,
    tid,
)


def get_autotune_config():
    # TODO(houqi.1993) keep len to be 1!!!
    # all machines should use exactly the same record for M/N/K/world_size
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
            num_stages=3,
            num_warps=4,
        ),
    ]


@triton.jit
def warp_prefix_sum_kernel(id, value):
    i = 1
    tl.static_print(value)

    while i < 32:
        val = __shfl_up_sync_i32(0xFFFFFFFF, value, i)
        # if tid(axis=0) < 5:
        #     tl.device_print("tid: ", tid(axis=0), val)
        if id >= i:
            value += val
        i = i * 2

    return value


@triton.jit
def save_warp_prefix_sum_kernel(output_ptr):
    id = tid(axis=0) % 32
    value = warp_prefix_sum_kernel(id, id)
    if tl.program_id(axis=0) == 0:
        if tid(axis=0) < 5:
            tl.device_print("tid, output, ntid: ", tid(axis=0), value)
        atomic_add(output_ptr + tid(axis=0), value, scope="gpu", semantic="relaxed")


def run_warp_prefix_sum():
    output = torch.zeros(1024, dtype=torch.int32, device="cuda")
    save_warp_prefix_sum_kernel[(1,)](output, num_warps=32)
    print(f"warp_prefix_sum output: {output.reshape(32, -1)}")


@triton.jit
def gemm_rs_thread_block_swizzle_kernel(
    # Matrix dimensions
    M,
    N,
    rank,
    world_size,
    threadblock_coord_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # threadblock swizzle: only support 1d-ring now.
    laneid = tid(axis=0) % 32
    lane_segment = (laneid + 1 + rank) % world_size
    m_per_rank = M // world_size
    m_start = m_per_rank * lane_segment
    m_end = m_per_rank * (lane_segment + 1) - 1
    tiled_m_start = m_start // BLOCK_SIZE_M
    tiled_m_end = m_end // BLOCK_SIZE_M

    laneid_prev_segment = (laneid - 1 + world_size) % world_size
    laneid_next_segment = (laneid + 1) % world_size
    prev_tiled_m_end = (m_start - 1) // BLOCK_SIZE_M
    next_tiled_m_start = (m_end + 1) // BLOCK_SIZE_M
    own_tiled_m_start = ((m_start == 0) or (prev_tiled_m_end != tiled_m_start)) or (
        laneid < laneid_prev_segment
    )
    own_tiled_m_end = ((m_end == M - 1) or (tiled_m_end != next_tiled_m_start)) or (
        laneid < laneid_next_segment
    )
    if not own_tiled_m_start:
        tiled_m_start += 1
    if not own_tiled_m_end:
        tiled_m_end -= 1
    segment_size_m = tl.maximum(0, tiled_m_end - tiled_m_start + 1)

    segment_size_m_accum = tl.where(
        tid(axis=0) < world_size, warp_prefix_sum_kernel(laneid, segment_size_m), 0
    )
    index_mask = __ballot_sync(0xFFFFFFFF, pid_m < segment_size_m_accum)
    segment = tl.math.ffs(index_mask) - 1
    segment_size_m_new = __shfl_sync_i32(0xFFFFFFFF, segment_size_m, segment)
    tiled_m_offset = tl.where(
        segment == 0, 0, __shfl_sync_i32(0xFFFFFFFF, segment_size_m_accum, segment - 1)
    )
    tiled_m_start_new = __shfl_sync_i32(0xFFFFFFFF, tiled_m_start, segment)

    # "pid tid pid_m lane_segment segment tiled_m_start tiled_m_end own_tiled_m_start own_tiled_m_end segment_size_m segment_size_m_new segment_size_m_accum tiled_m_offset tiled_m_start_new "
    if tid(axis=0) < world_size:
        tl.device_print(
            ">> ",
            pid,
            laneid,
            laneid_prev_segment,
            laneid_next_segment,
            pid_m,
            lane_segment,
            segment,
            tiled_m_start,
            tiled_m_end,
            own_tiled_m_start,
            own_tiled_m_end,
            segment_size_m,
            segment_size_m_new,
            segment_size_m_accum,
            tiled_m_offset,
            tiled_m_start_new,
        )

    pid_inner_segment = pid - tiled_m_offset * num_pid_n
    pid_n_new = pid_inner_segment // segment_size_m_new
    pid_m_new = pid_inner_segment % segment_size_m_new + tiled_m_start_new
    if tid(axis=0) == 0:
        atomic_add(threadblock_coord_ptr + pid * 2, pid_m_new, semantic="relaxed", scope="gpu")
        atomic_add(threadblock_coord_ptr + pid * 2 + 1, pid_n_new, semantic="relaxed", scope="gpu")


def run_gemm_rs_thread_block_swizzle(M, N, rank, world_size):
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    num_tiled_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_tiled_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    threadblock_coord = torch.zeros(
        num_tiled_m * num_tiled_n * 2,
        dtype=torch.int32,
        device="cuda",
    )
    grid = (num_tiled_m * num_tiled_n,)
    gemm_rs_thread_block_swizzle_kernel[grid](
        M, N, rank, world_size, threadblock_coord, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    print(threadblock_coord.reshape(num_tiled_m, num_tiled_n, 2))


if __name__ == "__main__":
    # run_warp_prefix_sum()
    run_gemm_rs_thread_block_swizzle(1024 - 8, 128 * 3, 0, 8)
    run_gemm_rs_thread_block_swizzle(1024, 128 * 3, 0, 8)
    run_gemm_rs_thread_block_swizzle(1024 + 8, 128 * 3, 0, 8)
    run_gemm_rs_thread_block_swizzle(64, 128 * 3, 0, 8)
