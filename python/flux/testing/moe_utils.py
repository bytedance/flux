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

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
import torch.distributions

try:
    import triton
    import triton.language as tl
except Exception as e:
    pass

import flux

INDEX_DTYPE = torch.int64

_DISTRIBUTION = {
    "xl-4in32-training": [
        0.025833333333333333,
        0.026041666666666668,
        0.0209375,
        0.02,
        0.051041666666666666,
        0.05364583333333333,
        0.018958333333333334,
        0.0328125,
        0.016979166666666667,
        0.030625,
        0.04822916666666666,
        0.05572916666666667,
        0.019166666666666665,
        0.036458333333333336,
        0.015833333333333335,
        0.0128125,
        0.029583333333333333,
        0.03770833333333334,
        0.04895833333333333,
        0.027916666666666666,
        0.036145833333333335,
        0.03197916666666667,
        0.022083333333333333,
        0.022395833333333334,
        0.0225,
        0.0278125,
        0.021875,
        0.04614583333333333,
        0.03239583333333333,
        0.0384375,
        0.019895833333333335,
        0.0490625,
    ]
}


@lru_cache(maxsize=8)
def _pre_distribution(key, device_index):
    return torch.tensor(_DISTRIBUTION[key], dtype=torch.float32).cuda(device_index)


@dataclass
class moe_gating_args:
    choosed_experts: torch.Tensor
    scatter_index: torch.Tensor
    gather_index: torch.Tensor
    topk_index: torch.Tensor
    splits_gpu: torch.Tensor


def calc_gather_index_stable(choosed_experts: torch.Tensor):
    _, index_choosed_experts = choosed_experts.flatten().sort(stable=True)
    gather_index = index_choosed_experts.to(torch.int32) // topk
    topk_index = torch.arange(0, topk, dtype=torch.int32, device="cuda").repeat(ntokens)[
        index_choosed_experts
    ]
    return gather_index, topk_index


def calc_gather_index(
    scatter_index: torch.Tensor,
    row_start: int,
    row_end: int,
    BLOCK_SIZE: int = 1024,
):
    @triton.jit
    def _kernel(
        scatter_index: torch.Tensor,
        gather_index: torch.Tensor,
        topk_index: torch.Tensor,
        ntokens: int,
        topk: int,
        row_start: int,
        row_end: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < ntokens * topk
        scatter_idx = tl.load(scatter_index + offset, mask=mask, other=-1)
        token_idx = offset // topk
        topk_idx = offset % topk
        token_idx_mask = (scatter_idx >= row_start) & (scatter_idx < row_end)
        tl.store(gather_index + scatter_idx - row_start, token_idx, mask=token_idx_mask)
        tl.store(topk_index + scatter_idx - row_start, topk_idx, mask=token_idx_mask)

    ntokens, topk = scatter_index.shape
    gather_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    topk_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    grid = lambda META: (triton.cdiv(ntokens * topk, META["BLOCK_SIZE"]),)
    _kernel[grid](
        scatter_index,
        gather_index,
        topk_index,
        ntokens,
        topk,
        row_start,
        row_end,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
    )
    return gather_index, topk_index


def calc_scatter_index_stable(choosed_experts):
    return (
        choosed_experts.flatten().argsort(stable=True).argsort().int().view(choosed_experts.shape)
    )


def calc_scatter_index_stable_sync(choosed_experts, splits_gpu):
    scatter_index = torch.ones_like(choosed_experts) * -1
    cumsum_gpu = splits_gpu.cumsum(dim=0) - splits_gpu
    E = splits_gpu.numel()
    for e in range(E):
        scatter_index[choosed_experts == e] = torch.arange(
            cumsum_gpu[e], cumsum_gpu[e] + splits_gpu[e], dtype=torch.int32, device="cuda"
        )  # this step is synchronized


@torch.no_grad()
def gen_moe_gating_args(
    E: int,
    topk: int,
    ntokens: int,
    drop_token_ratio: float = 0.0,
    weights: Optional[torch.Tensor] = None,
    generator: torch.Generator = None,
    stable=True,
):
    """
    Args:
        E: number of experts
        topk: number of experts to be selected
        ntokens: number of tokens
        drop_token_ratio: ratio of tokens to be dropped
        weights: weights[eid] * notkens for experts. shape [E].
            NOTE:
                * if weights is not set, use even distribution
                * weights only *guides* the token count by distribution, not precisely set the count
        generator: torch.Generator
        stable: whether to use stable moe gating
    Returns:
        moe_gating_args:
            choosed_experts: shape [ntokens, topk]
            scatter_index: shape [ntokens, topk]
            gather_index: shape [ntokens * topk]
            splits_gpu: shape [E]
            topk_index: shape [ntokens * topk]
    """
    if weights is not None:
        assert len(weights.shape) == 1 and weights.shape[0] == E, f"weights: {weights.shape}"
        if not weights.is_cuda:
            weights = weights.to("cuda")
    drop_token = drop_token_ratio > 0.0
    if weights is None:
        # use the very deterministic moe gating. don't respect drop_ratio.
        if drop_token:
            E = E + 1
        choosed_experts = (
            torch.arange(E, device="cuda", dtype=torch.int32)
            .repeat(ntokens, 1)
            .flatten()[: topk * ntokens]
            .reshape(ntokens, topk)
        )
    else:
        if drop_token:
            assert drop_token_ratio > 0.0 and drop_token_ratio < 1.0
            weights_with_drop = torch.zeros(E + 1, device="cuda", dtype=torch.float32)
            weights_with_drop[:E].copy_(weights)
            weights_with_drop[-1] = weights.sum() / (1 - drop_token_ratio) * drop_token_ratio
            weights = weights_with_drop
            E = E + 1

        choosed_experts = torch.multinomial(
            weights.repeat(ntokens, 1), topk, replacement=False, generator=generator
        ).to(torch.int32)

    splits_gpu = torch.bincount(choosed_experts.view(-1), minlength=E).to(
        torch.int32
    )  # this step is synchronized

    if stable:
        scatter_index = calc_scatter_index_stable(choosed_experts)
    else:
        scatter_index = flux.calc_scatter_index(choosed_experts, splits_gpu)

    try:
        gather_index, topk_index = calc_gather_index(scatter_index, 0, ntokens * topk)
    except Exception as e:
        assert stable == True, f"non-stable gather_index calculation requires triton"
        gather_index, topk_index = calc_gather_index_stable(choosed_experts)

    return moe_gating_args(
        choosed_experts=choosed_experts,
        scatter_index=scatter_index,
        gather_index=gather_index,
        topk_index=topk_index,
        splits_gpu=splits_gpu,
    )


@torch.no_grad()
def gate_func(
    token_num,
    num_experts,
    topk,
    dist: str = "random",
    generator: torch.Generator = None,
):
    if dist == "random":
        weights = torch.rand(num_experts, dtype=torch.float32, device="cuda", generator=generator)
    elif dist == "random_with_first_k_experts":
        weights = torch.rand(num_experts, dtype=torch.float32, device="cuda", generator=generator)
        weights[topk:].fill_(0)
    elif dist == "uniform":
        weights = torch.ones(num_experts, dtype=torch.float32, device="cuda")
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    weights = torch.ones(num_experts, dtype=torch.float32, device="cuda")
    args = gen_moe_gating_args(
        E=num_experts,
        topk=topk,
        ntokens=token_num,
        drop_token_ratio=0.0,
        weights=weights,
        generator=generator,
    )
    return (
        args.splits_gpu.to("cpu"),
        args.choosed_experts,
        args.gather_index,
        args.topk_index,
        args.scatter_index.flatten(),
    )


def _is_strictly_monotone_increasing(tensor: torch.Tensor) -> bool:
    return (tensor[1:] - tensor[:-1] > 0).all()


def is_gather_index_stable(gather_index: torch.Tensor, splits_gpu: torch.Tensor):
    """check if for all experts, gather_index is strictly monotone increasing"""
    # gather_index: [E * topk]
    # splits_gpu: [E]
    splits_cpu = splits_gpu.to("cpu")
    splits_cpu_cumsum = torch.zeros(splits_cpu.shape[0] + 1, dtype=torch.int32, device="cpu")
    splits_cpu_cumsum[1:].copy_(splits_cpu.cumsum(0))
    splits_cpu_cumsum = splits_cpu_cumsum.tolist()
    return all(
        [
            _is_strictly_monotone_increasing(gather_index[s:e])
            for s, e in zip(splits_cpu_cumsum[:-1], splits_cpu_cumsum[1:])
        ]
    )


if __name__ == "__main__":
    topk = 5
    E = 10
    ntokens = 10

    args = gen_moe_gating_args(E, topk, ntokens, drop_token_ratio=0.5)
    print(f"with drop ratio = 0.5")
    print(f"Arguments: {args}")
    print(f" {args.choosed_experts.dtype}")
    print(f" {args.gather_index.dtype}")
    print(f" {args.scatter_index.dtype}")
    print(f" {args.splits_gpu.dtype}")
    print(f" {args.topk_index.dtype}")
    print("\n\n\n")

    weights = torch.arange(E, device="cuda", dtype=torch.float32)
    args = gen_moe_gating_args(E, topk, ntokens, weights=weights)
    print(f"with weights: {weights}")
    print(f"Arguments: {args}")
    print("\n\n\n")

    args = gen_moe_gating_args(E, topk, ntokens)
    print(f"Arguments: {args}")

    ntokens = 10000
    args = gen_moe_gating_args(E, topk, ntokens, stable=False)
    print(f"Arguments: {args}")
    if is_gather_index_stable(args.gather_index, args.splits_gpu):
        print("gather_index is sorted")
    else:
        print("gather_index is not sorted")
