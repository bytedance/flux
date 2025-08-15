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
    __shfl_up_sync_i32,
    __syncthreads,
    arrive_inc,
    atomic_add,
    ld_acquire,
    red_release,
    tid,
    wait_eq,
)

int_dtypes = ["int8", "int16", "int32", "int64"]
uint_dtypes = ["uint8", "uint16", "uint32", "uint64"]
integral_dtypes = int_dtypes + uint_dtypes
float_dtypes = ["float16", "float32", "float64"]
dtypes = integral_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ["bfloat16"]
torch_float8_dtypes = ["float8_e4m3fn", "float8_e5m2"]
torch_dtypes = ["bool"] + int_dtypes + ["uint8"] + float_dtypes + ["bfloat16"]


def test_arrive_inc():

    @triton.jit
    def arrive_inc_kernel(barrier_ptr, scope: tl.constexpr):
        thread_idx = tid(axis=0)
        arrive_inc(barrier_ptr, thread_idx, 1, scope)
        # continous arrive_inc_sys will hang: don't know why
        # arrive_inc_sys(barrier_ptr, 1)
        # arrive_inc_sys(barrier_ptr + 1, 1)

    grid = (1, 1, 1)
    barrier = torch.zeros(1, dtype=torch.int32, device="cuda")
    for scope in ["gpu", "sys"]:
        barrier.zero_()
        arrive_inc_kernel[grid](barrier, scope)
        assert torch.allclose(barrier, torch.tensor([1], dtype=torch.int32, device="cuda")), barrier


def test_wait_eq():
    @triton.jit
    def wait_eq_kernel(barrier_ptr, scope: tl.constexpr):
        thread_idx = tid(axis=0)
        wait_eq(barrier_ptr, thread_idx, 1, scope)
        # continous wait_eq will hang: don't know why
        # wait_eq_sys(barrier_ptr, 1)
        # wait_eq_sys(barrier_ptr + 1, 1)

    grid = (1, 1, 1)
    barrier = torch.zeros(10, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    for scope in ["gpu", "sys"]:
        barrier.zero_()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100000000)
            barrier.fill_(1)
        wait_eq_kernel[grid](barrier, scope)
        torch.cuda.synchronize()


def test_red_release():

    @triton.jit
    def red_release_kernel(barrier_ptr, count, scope: tl.constexpr):
        __syncthreads()
        if tid(axis=0) < count:
            red_release(barrier_ptr + tid(axis=0), 1, scope)

    count = 3
    grid = (1, 1, 1)
    barrier = torch.zeros(count, dtype=torch.int32, device="cuda")
    for scope in ["gpu", "sys"]:
        barrier.zero_()
        red_release_kernel[grid](barrier, count, scope)
        assert torch.allclose(barrier, torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda"))


def test_ld_acquire():
    @triton.jit
    def ld_acquire_kernel(barrier_ptr, count, scope: tl.constexpr):
        if tid(axis=0) < count:
            while ld_acquire(barrier_ptr + tid(axis=0), scope) != 1:
                pass

    grid = (1, 1, 1)
    count = 3
    barrier = torch.zeros(count, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()

    for scope in ["gpu", "sys"]:
        barrier.fill_(0)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100000000)
            barrier.fill_(1)

        ld_acquire_kernel[grid](barrier, count, scope)
        torch.cuda.synchronize()


def test_atomic_add():
    @triton.jit
    def run_atomic_add_kernel(flag_ptr, count, scope: tl.constexpr, semantic: tl.constexpr):
        thread_idx = tid(axis=0)
        if thread_idx < count:
            for _ in range(thread_idx + 1):
                atomic_add(flag_ptr + thread_idx, 1, scope, semantic)
        __syncthreads()

    grid = (1, 1, 1)
    count = 3
    barrier = torch.zeros(count, dtype=torch.int32, device="cuda")
    for scope in ["gpu", "sys"]:
        for semantic in ["relaxed", "acquire", "release", "acq_rel"]:
            barrier.zero_()
            run_atomic_add_kernel[grid](barrier, count, scope, semantic, num_warps=1)
            assert torch.allclose(
                barrier, torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
            )


def test_shfl_sync():
    @triton.jit
    def warp_prefix_sum_kernel(input, output):
        i = 1
        thread_idx = tl.cast(tid(axis=0), tl.int32)
        value = tl.load(input + thread_idx)
        laneid = thread_idx % 32
        while i < 32:
            val = __shfl_up_sync_i32(0xFFFFFFFF, value, i)
            if laneid >= i:
                value += val
            i = i * 2

        atomic_add(output + thread_idx, value, "gpu", "relaxed")

    N = 1024
    # N = 32
    in_tensor = torch.randint(0, 255, (N,), dtype=torch.int32, device="cuda")
    # in_tensor = torch.ones((N,), dtype=torch.int32, device="cuda")
    out_tensor = torch.zeros_like(in_tensor)
    gt_tensor = torch.zeros_like(in_tensor)
    torch.cumsum(in_tensor.reshape(-1, 32), dim=1, out=gt_tensor)
    gt_tensor = gt_tensor.reshape(N)
    grid = (1, 1, 1)
    warp_prefix_sum_kernel[grid](in_tensor, out_tensor, num_warps=N // 32)
    assert torch.allclose(gt_tensor, out_tensor), (gt_tensor, out_tensor, in_tensor)


def test_ballot_sync():
    @triton.jit
    def run_ballot_sync_kernel(tensor, count, target, output):
        thread_idx = tid(axis=0)
        val = tl.load(tensor + thread_idx) if thread_idx < count else -1
        out = __ballot_sync(0xFFFFFFFF, val == target)
        __syncthreads()
        tl.store(output, out)

    in_tensor = torch.arange(32, dtype=torch.int32, device="cuda")
    out_tensor = torch.zeros((1,), dtype=torch.uint32, device="cuda")
    for target in range(8):
        run_ballot_sync_kernel[(1,)](in_tensor, 8, target, out_tensor)
        assert int(out_tensor[0]) == (1 << target)


def test_ld():
    pass


def test_sync_threads():
    @triton.jit
    def sync_threads_kernel():
        __syncthreads()

    sync_threads_kernel[(1, 1, 1)]()


if __name__ == "__main__":

    test_ld_acquire()

    test_atomic_add()

    test_red_release()

    test_sync_threads()

    test_wait_eq()

    test_arrive_inc()

    test_shfl_sync()

    test_ballot_sync()
