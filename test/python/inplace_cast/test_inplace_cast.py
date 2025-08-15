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

import flux


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1048580, 1048572, 1048573, 1048575, 1048571, 8388604, 8388603, 8388605],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import flux_ths_pybind as ths

    # Test passed on (1048576 1048580 1048572 1048573 1048575 1048571 8388604 8388603 8388605)

    # Get the maximum data size from args.sizes
    max_data_size = max(args.sizes)

    ## Test inplace_cast_fp32_to_bf16 API
    for data_size in args.sizes:
        test_input = torch.rand(data_size, dtype=torch.float32).cuda()
        ref_output = test_input.to(torch.bfloat16)
        ths.inplace_cast_fp32_to_bf16(test_input)
        test_output = test_input.view(torch.bfloat16)
        test_output = torch.narrow(test_output, 0, 0, data_size)
        flux.torch_allclose(test_output, ref_output, 1e-5, 1e-8)

    ## Test InplaceCast class: create object outside loop
    inplace_cast_op = flux.InplaceCast(max_data_size)
    for data_size in args.sizes:
        test_data = torch.rand(data_size, dtype=torch.float32).cuda()
        golden = test_data.to(torch.bfloat16)
        inplace_cast_op.from_fp32_to_bf16(test_data)
        test_data_bf16 = test_data.view(torch.bfloat16)
        test_data_bf16 = torch.narrow(test_data_bf16, 0, 0, data_size)
        flux.torch_allclose(test_data_bf16, golden, 1e-5, 1e-8)

    ## Test InplaceCast class: create object inside loop
    for data_size in args.sizes:
        inplace_cast_op = flux.InplaceCast(data_size)
        test_data = torch.rand(data_size, dtype=torch.float32).cuda()
        golden = test_data.to(torch.bfloat16)
        inplace_cast_op.from_fp32_to_bf16(test_data)
        test_data_bf16 = test_data.view(torch.bfloat16)
        test_data_bf16 = torch.narrow(test_data_bf16, 0, 0, data_size)
        flux.torch_allclose(test_data_bf16, golden, 1e-5, 1e-8)
