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
import torch.distributed

import flux
from flux.testing import all_gather_into_tensor_with_fp8, gen_moe_gating_args, matmul_int8
from flux.testing.utils import generate_data


class MoeMlp1Ctx:
    def __init__(
        self,
        TP_GROUP,
        EP_GROUP,
        b: int,
        s: int,
        h: int,  # K
        ffn_size: int,
        nexperts: int,
        topk: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        dist: str,
        fast_accum: bool,
        weight_groups: int,
        drop_token: bool,
        debug: bool = False,
        generator: torch.Generator = None,  # torch random generator
        stable: bool = True,
    ) -> None:
        self.b = b
        self.s = s
        self.h = h
        self.ffn_size = ffn_size
        self.nexperts = nexperts
        self.topk = topk
        self.ntokens = b * s
        self.fast_accum = fast_accum
        self.weight_groups = weight_groups
        self.tp_rank = TP_GROUP.rank()
        self.tp_size = TP_GROUP.size()
        self.ep_rank = EP_GROUP.rank()
        self.ep_size = EP_GROUP.size()
        self.ffn_tp_size = self.tp_size // self.ep_size
        self.nexperts_ep = self.nexperts // self.ep_size
        self.drop_token = drop_token
        assert self.nexperts % self.ep_size == 0

        assert self.ffn_size % self.ffn_tp_size == 0
        self.ffn_size_shard = ffn_size // self.ffn_tp_size
        assert self.ntokens % self.tp_size == 0
        self.ntokens_shard = self.ntokens // self.tp_size

        device = torch.cuda.current_device()

        is_s8_dequant = input_dtype == torch.int8
        is_fp8 = flux.is_fp8_dtype(input_dtype)

        if dist == "uniform":
            weights = None
        elif dist == "random_uniform":
            weights = torch.ones(self.nexperts, device=device, dtype=torch.float32)
        else:
            weights = torch.rand(self.nexperts, device=device, dtype=torch.float32)
            if dist == "random_with_first_k_experts":
                weights[self.topk :].fill_(0)

        moe_gating_args = gen_moe_gating_args(
            nexperts,
            topk,
            self.ntokens,
            0.1 if drop_token else 0.0,
            stable=stable,
            weights=weights,
            generator=generator,
        )

        self.splits_gpu = moe_gating_args.splits_gpu
        self.splits_cpu = moe_gating_args.splits_gpu.to("cpu")
        self.scatter_index = moe_gating_args.scatter_index
        self.gather_index = moe_gating_args.gather_index

        self.nrows_ep = int(
            torch.sum(
                self.splits_cpu[
                    self.nexperts_ep * self.ep_rank : self.nexperts_ep * (self.ep_rank + 1)
                ]
            )
        )

        N, K = self.ffn_size_shard, h
        E_this_ep = self.nexperts_ep

        if is_s8_dequant:
            scale_value = 127
        elif is_fp8:
            scale_value = 0.1
        else:
            scale_value = 0.01 * (self.tp_rank + 1)

        data_config = [
            ((self.ntokens_shard, K), input_dtype, (scale_value, 0)),  # input_shard
            ((self.ntokens, K), input_dtype, (1, 0)),  # input_full
            ((self.ntokens * topk, K), input_dtype, (1, 0)),  # scatter_inputs
        ]
        self.inputs_shard, self.inputs, self.scatter_inputs = next(generate_data(data_config))

        data_config = [
            ((E_this_ep, N, K), input_dtype, (scale_value, 0)),  # weights
            ((self.nrows_ep, N), output_dtype, (0, 0)),  # outputs
            ((E_this_ep,), torch.float32, (1.0, 0)),  # output_scale
        ]
        generator = generate_data(data_config)
        self.weights, self.outputs, self.output_scale = [
            *zip(*[list(next(generator)) for _ in range(weight_groups)])
        ]
        if is_s8_dequant:
            data_config = [
                ((E_this_ep, 1, N), torch.float32, (1.0, 0)),  # bias
                ((self.ntokens_shard), torch.float32, (1.0, 0)),  # input_scale_shard
                ((E_this_ep, 1, N), torch.float32, (1.0, 0)),  # weight_scale
                ((self.ntokens, 1), torch.float32, (1.0, 0)),  # input_scale
                ((self.ntokens * topk, 1), torch.float32, (1.0, 0)),  # scatter_input_scale
            ]
            generator = generate_data(data_config)
            (
                self.bias,
                self.input_scale,
                self.weight_scale,
                self.full_input_scale,
                self.scatter_input_scale,
            ) = [*zip(*[list(next(generator)) for _ in range(weight_groups)])]
        else:
            (
                self.bias,
                self.input_scale,
                self.weight_scale,
                self.full_input_scale,
                self.scatter_input_scale,
            ) = (None, None, None, None, None)

        if debug:
            self._fill_debug()

        torch.cuda.synchronize()
        self.is_s8 = is_s8_dequant
        self.is_fp8 = is_fp8

    def _fill_debug(self):
        input_dtype = self.inputs_shard.dtype
        is_s8_dequant = input_dtype == torch.int8
        is_fp8 = flux.is_fp8_dtype(input_dtype)
        rank = self.tp_rank
        with flux.with_torch_deterministic(not is_fp8):
            self.inputs_shard.zero_()
            self.inputs_shard[:, 0] = (
                torch.arange(rank * self.ntokens_shard + 1, (rank + 1) * self.ntokens_shard + 1)
                / 1e2
            )
            # self.inputs_shard.fill_(1 / 32)
            E, _, _ = self.weights[0].shape
            for w in self.weights:
                for e in range(E):
                    w[e, :, :].fill_(e + 1)
            [x.fill_(1) for x in self.output_scale]
            if is_s8_dequant:
                self.bias[0].fill_(0)
                self.input_scale[0].fill_(1)
                self.weight_scale[0].fill_(1)

    def clear_outputs(self):
        for i in range(self.weight_groups):
            self.outputs[i].fill_(0.0)

    def get_outputs_clone(self):
        return [out.clone() for out in self.outputs]


class MoeAgScatterWithTorch(object):
    @staticmethod
    def comm_impl(ctx, TP_GROUP):
        all_gather_into_tensor_with_fp8(ctx.inputs, ctx.inputs_shard, group=TP_GROUP)
        if ctx.is_s8:
            if ctx.input_scale is not None:
                for i in range(ctx.weight_groups):
                    torch.distributed.all_gather_into_tensor(
                        ctx.full_input_scale[i], ctx.input_scale[i], group=TP_GROUP
                    )

    @staticmethod
    def scatter_impl(ctx):
        if flux.is_fp8_dtype(ctx.inputs.dtype):
            ctx.scatter_inputs.view(torch.uint8).copy_(
                torch.index_select(ctx.inputs.view(torch.uint8), dim=0, index=ctx.gather_index)
            )
        else:
            ctx.scatter_inputs.copy_(torch.index_select(ctx.inputs, dim=0, index=ctx.gather_index))

        if ctx.is_s8 and ctx.input_scale is not None:
            for n in range(ctx.weight_groups):
                ctx.scatter_input_scale[n].copy_(
                    torch.index_select(ctx.full_input_scale[n], dim=0, index=ctx.gather_index)
                )

    @staticmethod
    def gemm_impl(ctx, gemm_only_op):
        offset = 0
        expert_id_offset = ctx.nexperts_ep * ctx.ep_rank
        input_offset = torch.sum(ctx.splits_cpu[:expert_id_offset])
        for exp_id in range(ctx.nexperts_ep):
            nxt_offset = offset + ctx.splits_cpu[exp_id + expert_id_offset]
            if nxt_offset - offset > 0:
                exp_input = ctx.scatter_inputs[offset + input_offset : nxt_offset + input_offset]
                for i in range(ctx.weight_groups):
                    out = ctx.outputs[i][offset:nxt_offset]
                    if flux.is_fp8_dtype(exp_input.dtype):
                        gemm_only_op.forward(
                            exp_input,
                            ctx.weights[i][exp_id],
                            output_buf=out,
                            fast_accum=ctx.fast_accum,
                        )
                    elif ctx.is_s8:
                        input_scale = ctx.scatter_input_scale[i][
                            offset + input_offset : nxt_offset + input_offset, :
                        ]
                        accum = matmul_int8(exp_input, ctx.weights[i][exp_id].t())
                        out.copy_(
                            (
                                input_scale * ctx.weight_scale[i][exp_id] * accum.to(torch.float32)
                            ).to(torch.bfloat16)
                        )
                    else:
                        torch.matmul(
                            exp_input,
                            ctx.weights[i][exp_id].t(),
                            out=out,
                        )

                    # TODO(houqi.1993) xperf_gpt has no output_scale. so don't know the order of output_scale and bias, leave it as bias first.
                    if ctx.bias is not None:
                        out.add_(ctx.bias[i][exp_id])
                    if not ctx.is_s8:
                        out.mul_(ctx.output_scale[i][exp_id])

            offset = nxt_offset
