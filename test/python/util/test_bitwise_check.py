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

import argparse

import torch
import torch.distributed

from flux.testing import DTYPE_MAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=12288)
    parser.add_argument("-B", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")

    return parser.parse_args()


def recover_from_bcsr(ori_t, block_h, block_w):
    assert ori_t.size(0) % block_h == 0
    assert ori_t.size(1) % block_w == 0
    M = ori_t.size(0)
    N = ori_t.size(1)
    BM = block_h
    BN = block_w
    new_out = torch.empty_like(ori_t)
    # tmp_t = torch.em
    tmp_t = ori_t.flatten().view(M // BM, N // BN, BM, BN)
    # if(RANK==0):
    #     import pdb; pdb.set_trace()
    for bmi in range(ori_t.size(0) // block_h):
        for bni in range(ori_t.size(1) // block_w):
            m_start = bmi * BM
            m_end = m_start + BM
            n_start = bni * BN
            n_end = n_start + BN
            new_out[m_start:m_end, n_start:n_end] = tmp_t[bmi, bni]
    return new_out


if __name__ == "__main__":
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    input = torch.rand(args.B, args.M, args.N).cuda().to(dtype)
    # input[:, 0,:] = 1
    output = torch.zeros(args.M, args.N).cuda().to(dtype)
    # red_output = torch.zeros(args.M, args.N).cuda().to(dtype)
    ref_out = torch.sum(input, dim=0)
    ref_out = recover_from_bcsr(ref_out, 128, 128)
    import flux_ths_pybind as ths

    ths.bsr_reduce(input, output, 128, 128)
    ret = ths.bitwise_check(ref_out, output)
    assert ret == True
    print("Bitwise check passed!")
