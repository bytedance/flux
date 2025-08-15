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


def torch_pre_attn_all_to_all_transpose(
    sp_group, input, bs, seq_len, nh, head_dim, seq_lens_cpu=None
):
    if seq_lens_cpu != None:
        raise NotImplementedError("a2a transpose does not support dp.")
    local_nh = nh // sp_group.size()
    local_seq_len = seq_len // sp_group.size()
    input = input.reshape(bs, local_seq_len, nh, head_dim)
    a2a_buffer = torch.empty(
        (seq_len, bs, local_nh, head_dim),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    input_before_a2a = input.permute(2, 1, 0, 3).contiguous()  # [nh, local_seq_len, bs, hd]
    torch.distributed.all_to_all_single(a2a_buffer, input_before_a2a, group=sp_group)

    a2a_buffer = (
        a2a_buffer.reshape(sp_group.size(), local_nh, local_seq_len, bs, head_dim)
        .permute(3, 1, 0, 2, 4)
        .reshape(bs, local_nh, seq_len, head_dim)
    )
    return [a2a_buffer]


def torch_pre_attn_qkv_pack_a2a(sp_group, input, bs, seq_len, nh, head_dim, gqa, seq_lens_cpu=None):
    world_size = sp_group.size()
    rank = sp_group.rank()
    local_seq_len = (
        seq_len // sp_group.size() if seq_lens_cpu == None else seq_lens_cpu[rank].item()
    )
    local_nh = nh // sp_group.size()
    input = input.reshape(bs, local_seq_len, nh, head_dim)
    local_q_nh = local_nh // (gqa + 2) * gqa
    local_k_nh = local_nh // (gqa + 2)
    local_v_nh = local_k_nh
    q_input = input[:, :, : local_q_nh * world_size, :].contiguous()
    k_input = input[
        :, :, local_q_nh * world_size : (local_q_nh + local_k_nh) * world_size, :
    ].contiguous()
    v_input = input[:, :, (local_q_nh + local_k_nh) * world_size :, :].contiguous()

    def _a2a(a2a_tensor):
        if seq_lens_cpu == None:
            a2a_input = a2a_tensor.permute(2, 1, 0, 3).contiguous()  # [nh, local_seq_len, bs, hd]
            a2a_nh, a2a_local_seq_len, a2a_bs, a2a_hd = a2a_input.shape
            a2a_buffer = torch.empty(
                (world_size, a2a_nh // world_size, a2a_local_seq_len, a2a_bs, a2a_hd),
                dtype=a2a_input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            torch.distributed.all_to_all_single(a2a_buffer, a2a_input, group=sp_group)
            a2a_buffer = (
                a2a_buffer.permute(3, 0, 2, 1, 4)
                .reshape(a2a_bs, a2a_local_seq_len * world_size, a2a_nh // world_size, a2a_hd)
                .contiguous()
            )
            return a2a_buffer
        else:
            a2a_nh = a2a_tensor.shape[2]
            a2a_local_nh = a2a_nh // world_size
            a2a_input = (
                a2a_tensor.permute(2, 1, 0, 3).reshape(-1, bs, head_dim).contiguous()
            )  # [nh * local_seq_len , bs, hd]
            _, a2a_bs, a2a_hd = a2a_input.shape
            sum_seq_len = seq_lens_cpu.sum().item()
            a2a_buffer = torch.empty(
                (a2a_local_nh * sum_seq_len, a2a_bs, a2a_hd),
                dtype=a2a_input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            output_splits = [val * a2a_local_nh for val in seq_lens_cpu.tolist()]
            input_splits = [output_splits[rank] for i in range(world_size)]
            torch.distributed.all_to_all_single(
                a2a_buffer, a2a_input, output_splits, input_splits, group=sp_group
            )
            tensor_list = []
            start = 0
            for i in range(world_size):
                cur_slice = a2a_buffer[start : start + output_splits[i], :, :].reshape(
                    a2a_local_nh, -1, a2a_bs, a2a_hd
                )
                start += output_splits[i]
                tensor_list.append(cur_slice)
            a2a_buffer = torch.cat(
                tensor_list, dim=1
            )  # [a2a_local_nh, sum_seq_len, a2a_bs, a2a_hd]
            a2a_buffer = a2a_buffer.permute(2, 1, 0, 3).contiguous()
            return a2a_buffer

    q = _a2a(q_input)
    k = _a2a(k_input)
    v = _a2a(v_input)
    return [q, k, v]


def torch_post_attn_all_to_all_transpose(sp_group, input, a2a_only, is_dp, seq_lens_cpu=None):
    if not a2a_only:
        bs, local_nh, seq_len, hd = input.shape
    else:
        bs, seq_len, local_nh, hd = input.shape
    local_seq_len = seq_len // sp_group.size()
    hidden_dim = local_nh * hd * sp_group.size()

    if is_dp:
        local_seq_len = seq_lens_cpu[sp_group.rank()].item()

    # All to all input tensors from all gpus
    input_after_a2a = torch.zeros(
        (local_seq_len * sp_group.size(), bs, local_nh, hd),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    if is_dp:
        input_before_a2a = input.permute(1, 0, 2, 3).contiguous()  # [seq_len, bs, local_nh, hd]
        output_splits = [local_seq_len for i in range(sp_group.size())]
        input_splits = seq_lens_cpu.tolist()
        torch.distributed.all_to_all_single(
            input_after_a2a, input_before_a2a, output_splits, input_splits, group=sp_group
        )
        gemm_input = (
            input_after_a2a.reshape(sp_group.size(), local_seq_len, bs, local_nh, hd)
            .permute(2, 1, 0, 3, 4)
            .reshape(bs, local_seq_len, hidden_dim)
        )
    else:
        if not a2a_only:
            input_before_a2a = input.permute(2, 0, 1, 3).contiguous()
            torch.distributed.all_to_all_single(input_after_a2a, input_before_a2a, group=sp_group)
            gemm_input = (
                input_after_a2a.reshape(sp_group.size(), local_seq_len, bs, local_nh, hd)
                .permute(2, 1, 0, 3, 4)
                .reshape(bs, local_seq_len, hidden_dim)
            )
        else:
            input_before_a2a = input.permute(1, 0, 2, 3).contiguous()  # [seq_len, bs, local_nh, hd]
            torch.distributed.all_to_all_single(input_after_a2a, input_before_a2a, group=sp_group)
            gemm_input = (
                input_after_a2a.reshape(sp_group.size(), local_seq_len, bs, local_nh, hd)
                .permute(2, 1, 0, 3, 4)
                .reshape(bs, local_seq_len, hidden_dim)
            )
    return gemm_input
