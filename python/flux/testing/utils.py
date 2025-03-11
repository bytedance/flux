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

import datetime
import os
import random
from typing import Callable, List, Tuple, Union, Sequence

import numpy as np
import torch
import torch.distributed

import flux

_TP_LOCAL_GROUP = None
_TP_GROUP = None

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

RING_MODE_MAP = {
    "auto": None,
    "all2all": flux.AGRingMode.All2All,
    "ring1d": flux.AGRingMode.Ring1D,
    "ring2d": flux.AGRingMode.Ring2D,
}


def init_seed(seed=0):
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)


def NNODES() -> int:
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    return WORLD_SIZE // LOCAL_WORLD_SIZE


def initialize_distributed():
    global _TP_GROUP
    assert _TP_GROUP is None, "TP_GROUP has already been initialized"

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", world_size=WORLD_SIZE, rank=RANK, timeout=datetime.timedelta(seconds=1800)
    )
    assert torch.distributed.is_initialized()
    # use all ranks as tp group
    _TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    init_seed(seed=RANK)
    flux.init_flux_shm(_TP_GROUP)
    torch.cuda.synchronize()
    return _TP_GROUP


def TP_GROUP() -> torch.distributed.ProcessGroup:
    global _TP_GROUP
    assert _TP_GROUP is not None, "TP_GROUP has not been initialized"
    return _TP_GROUP


def is_local_tp_group_initialized():
    return _TP_LOCAL_GROUP is not None


def init_local_groups():
    global _TP_LOCAL_GROUP
    assert _TP_LOCAL_GROUP is None, "TP_LOCAL_GROUP has already been initialized"
    assert _TP_GROUP is not None, "TP_GROUP has not been initialized"
    nnodes = NNODES()
    RANK = _TP_GROUP.rank()
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if nnodes == 1:
        _TP_LOCAL_GROUP = _TP_GROUP
    else:
        for n in range(nnodes):
            ranks = list(range(LOCAL_WORLD_SIZE * n, LOCAL_WORLD_SIZE * (n + 1)))
            pg = torch.distributed.new_group(
                ranks=ranks,
                backend="nccl",
            )
            if RANK in ranks:
                _TP_LOCAL_GROUP = pg
    assert LOCAL_RANK == RANK % LOCAL_WORLD_SIZE
    assert _TP_LOCAL_GROUP.rank() == RANK % LOCAL_WORLD_SIZE
    assert _TP_LOCAL_GROUP.size() == LOCAL_WORLD_SIZE
    return _TP_LOCAL_GROUP


def _async_barrier():
    barrier_tensor = torch.tensor([1], device="cuda")
    torch.distributed.all_reduce(barrier_tensor, op=torch.distributed.ReduceOp.MAX, async_op=False)


def _make_tensor(
    shape: List[Union[int, Callable[[], int]]],
    dtype: torch.dtype,
    init_args: Union[Tuple[float, float], Tuple[int, int]],
    device: str = "cuda",
):
    """
    rand() * scale + bias
    randint(-scale, scale) + bias
    """
    if isinstance(shape, Sequence):
        shape = tuple([x() if isinstance(x, Callable) else x for x in shape])
    elif isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, Callable):
        shape = shape()
    else:
        raise ValueError(f"unsupported shape {shape}")

    scale, bias = init_args
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        out = (torch.rand(shape, dtype=dtype, device=device) * 2 - 1) * scale + bias
    elif dtype == torch.int8:
        out = torch.randint(-scale, scale, shape, dtype=torch.int8, device=device)
        out = out + bias
    elif flux.util.is_fp8_dtype(dtype):
        out = (torch.rand(shape, dtype=torch.float16, device=device) * 2 - 1) * scale + bias
        with flux.util.with_torch_deterministic(False):
            out = out.to(dtype)
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    return out


def generate_data(configs):
    while True:
        yield (_make_tensor(*args) if args else None for args in configs)


def run_perf(expname, warmups, iters, func, sync_per_iter=False):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for n in range(warmups + iters):
        if n >= warmups:
            start_events[n - warmups].record()
            func(iter=n)
            stop_events[n - warmups].record()
        else:
            func(iter=n)
        if sync_per_iter:
            _async_barrier()
    [e.synchronize() for e in stop_events]
    elapsed_time_avg = (
        sum(
            [
                start_event.elapsed_time(stop_event)
                for start_event, stop_event in zip(start_events, stop_events)
            ]
        )
        / iters
    )
    print(f"exp: {expname}, avg: {elapsed_time_avg:0.3f} ms/iter")
    return elapsed_time_avg


def zeros_with_fp8(*args, **kwargs):
    with flux.util.with_torch_deterministic(False):
        return torch.zeros(*args, **kwargs)


def ones_with_fp8(*args, **kwargs):
    with flux.util.with_torch_deterministic(False):
        return torch.ones(*args, **kwargs)


def empty_non_determinisitc(*args, **kwargs):
    with flux.util.with_torch_deterministic(False):
        return torch.empty(*args, **kwargs)


def rand_with_fp8(*args, **kwargs):
    dtype = kwargs.get("dtype", None)
    kwargs.pop("dtype")
    if flux.util.is_fp8_dtype(dtype):
        with flux.util.with_torch_deterministic(False):
            return torch.rand(*args, dtype=torch.bfloat16, **kwargs).to(dtype)
    else:
        return torch.rand(*args, **kwargs)


def clone_with_fp8(tensor: torch.Tensor):
    with flux.util.with_torch_deterministic(False):
        return tensor.clone()


def all_gather_into_tensor_with_fp8(output_tensor, input_tensor, group=None, async_op=False):
    if flux.util.is_fp8_dtype(input_tensor.dtype):
        return torch.distributed.all_gather_into_tensor(
            output_tensor.view(torch.uint8),
            input_tensor.view(torch.uint8),
            group=group,
            async_op=async_op,
        )
    return torch.distributed.all_gather_into_tensor(
        output_tensor, input_tensor, group=group, async_op=async_op
    )


def matmul_int8(a, b):
    """
    torch._int_mm requires A.size(0) needs to be greater than 16
    """
    M, _ = a.shape
    if M <= 16:
        return torch._int_mm(torch.nn.functional.pad(a, (0, 0, 0, 32 - M)), b)[:M, :]
    return torch._int_mm(a, b)


def bitwise_eq(x: torch.Tensor, y: torch.Tensor):
    if x.shape != y.shape:
        return False
    if x.dtype != y.dtype:
        return False

    return torch.equal(x.view(torch.int8), y.view(torch.int8))


__all__ = [
    "DTYPE_MAP",
    "RING_MODE_MAP",
    "TP_GROUP",
    "NNODES",
    "init_seed",
    "initialize_distributed",
    "is_local_tp_group_initialized",
    "init_local_groups",
    "_async_barrier",
    "_make_tensor",
    "generate_data",
    "run_perf",
    "zeros_with_fp8",
    "ones_with_fp8",
    "rand_with_fp8",
    "empty_non_determinisitc",
    "all_gather_into_tensor_with_fp8",
    "matmul_int8",
    "bitwise_eq",
    "clone_with_fp8",
]
