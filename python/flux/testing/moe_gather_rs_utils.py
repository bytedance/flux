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
from typing import List, Union

import torch
import torch.distributed

import flux
from flux.testing import matmul_int8


def moe_gather_rs_forward_torch(
    TP_GROUP,
    M: int,
    eid_start: int,
    ep_rank_m_start: int,
    ep_rank_m_end: int,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    weights: Union[torch.Tensor, List[torch.Tensor]],
    split_cpu: torch.Tensor,
    token_index: torch.Tensor,
    topk_index: torch.Tensor,
    topk: int,
    input_scales: Union[torch.Tensor, List[torch.Tensor]],
    weight_scales: Union[torch.Tensor, List[torch.Tensor]],
    output_vec_scales: Union[torch.Tensor, List[torch.Tensor]],
    do_all_reduce: bool = False,
    fast_acc: bool = False,
):
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
        weights = [weights]
        input_scales = [input_scales]
        weight_scales = [weight_scales]
        output_vec_scales = [output_vec_scales]

    input0 = inputs[0]
    input_dtype = input0.dtype
    _, N, _ = weights[0].shape
    is_fp8 = flux.util.is_fp8_dtype(input_dtype)
    is_s8 = input_dtype == torch.int8
    output_type = torch.bfloat16 if is_fp8 or is_s8 else input_dtype

    # convert to list first
    full_output = torch.zeros((M, N), dtype=output_type, device=input0.device)
    for input_gid in range(len(inputs)):
        input = inputs[input_gid]
        weight = weights[input_gid]
        input_scale = input_scales[input_gid]
        weight_scale = weight_scales[input_gid]
        output_vec_scale = output_vec_scales[input_gid]

        acc = 0
        output_list = []

        gemm_only_op = flux.GemmOnly(
            input.dtype, output_type, use_fp8_gemm=flux.util.get_arch() < 90 and is_fp8
        )

        for exp_id in range(weight.size(0)):
            exp_w = weight[exp_id]
            Mi = split_cpu[exp_id + eid_start]
            exp_input = input[acc : acc + Mi]
            if not is_s8:
                scale_v = weight_scale[exp_id].item() * input_scale[0].item()
            else:
                scale_v = weight_scale[exp_id, None, :] * input_scale[acc : acc + Mi, None]
            acc += Mi
            if is_fp8:
                output_buf = torch.empty(Mi, N, dtype=torch.bfloat16).to(input.device)
                gemm_only_op.forward(exp_input, exp_w, output_buf=output_buf, fast_accum=fast_acc)
                output_list.append(scale_v * output_buf)
            elif is_s8 and Mi > 0:
                output = matmul_int8(exp_input, exp_w.t()).to(torch.float32) * scale_v
                output_list.append(output)
            else:
                output_list.append(scale_v * torch.matmul(exp_input, exp_w.t()))
        # M N
        output = torch.concat(output_list)
        # print(output.size())
        # print(output_vec_scale.size())
        assert output.size(0) == output_vec_scale.size(0)
        output = (output * output_vec_scale.unsqueeze(1)).to(output_type)

        new_index = (
            topk * token_index[ep_rank_m_start:ep_rank_m_end]
            + topk_index[ep_rank_m_start:ep_rank_m_end]
        )

        output1 = torch.zeros_like(full_output)
        output1[new_index] = output
        full_output += output1
    topk_reduce = full_output.view((full_output.size(0) // topk, topk, full_output.size(1))).sum(1)
    if do_all_reduce:
        torch.distributed.all_reduce(
            topk_reduce,
            op=torch.distributed.ReduceOp.SUM,
            group=TP_GROUP,
        )
        output2 = topk_reduce
    else:
        output2 = torch.zeros(
            (full_output.size(0) // TP_GROUP.size() // topk, full_output.size(1)),
            dtype=topk_reduce.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        torch.distributed.reduce_scatter_tensor(output2, topk_reduce, group=TP_GROUP)
    return output2
